import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from cytomine.models import Job
from neubiaswg5 import CLASS_PIXCLA
from neubiaswg5.helpers import get_discipline, NeubiasJob, prepare_data, upload_data, upload_metrics
from neubiaswg5.helpers.data_upload import imwrite, imread

from unet import UNet


def normalize(x):
    return x / 255


def predict_img(net, full_img, scale_factor=0.5, out_threshold=0.5):
    net.eval()
    height, width, channel = full_img.shape
    img = cv2.resize(full_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    img = normalize(img)
    img = np.transpose(img, axes=[2, 0, 1])
    x = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        y = net(x)
        proba = y.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((height, width)),
                transforms.ToTensor()
            ]
        )

        proba = tf(proba.cpu())
        mask_np = proba.squeeze().cpu().numpy()

    return mask_np > out_threshold


def load_model(filepath):
    net = UNet(n_channels=3, n_classes=1)
    net.cpu()
    net.load_state_dict(torch.load(filepath, map_location='cpu'))
    return net


def main(argv):
    with NeubiasJob.from_cli(argv) as nj:
        problem_cls = get_discipline(nj, default=CLASS_PIXCLA)
        is_2d = True

        nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, **nj.flags)

        # 2. Call the image analysis workflow
        nj.job.update(progress=10, statusComment="Load model...")
        net = load_model("/app/CP58_dice_0.9373_loss_0.0265.pth")

        for in_image in nj.monitor(in_images, start=20, end=75, period=0.05, prefix="Apply UNet to input images"):
            img = imread(in_image.filepath, is_2d=is_2d)

            mask = predict_img(
                net=net, full_img=img,
                scale_factor=0.5,  # value used at training
                out_threshold=nj.parameters.threshold
            )

            imwrite(
                path=os.path.join(out_path, in_image.filename),
                image=mask.astype(np.uint8),
                is_2d=is_2d
            )

        # 4. Create and upload annotations
        nj.job.update(progress=70, statusComment="Uploading extracted annotation...")
        upload_data(problem_cls, nj, in_images, out_path, **nj.flags, is_2d=is_2d, monitor_params={
            "start": 70, "end": 90, "period": 0.1
        })

        # 5. Compute and upload the metrics
        nj.job.update(progress=90, statusComment="Computing and uploading metrics (if necessary)...")
        upload_metrics(problem_cls, nj, in_images, gt_path, out_path, tmp_path, **nj.flags)

        # 6. End the job
        nj.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])

