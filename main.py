import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress

from mmdet.apis import inference_detector, init_detector
from lane import *
from mmdet.registry import VISUALIZERS

config_file = 'config.py'
checkpoint_file = 'models/epoch_60.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

def process_file_img(filename):
    img = mmcv.imread(rf"static/videos/input/{filename}",channel_order='rgb')
    result = inference_detector(model, img)
    out_file=rf"videos/output/output_{filename}"
    lines = lane(img)
    img = plot_lines(img,lines)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    # show the results
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt = None,
        wait_time=0,
        out_file=rf"static/videos/output/output_{filename}"
    )
    return out_file

def process_file(filename):
    video=rf"static/videos/input/{filename}"
    processed_filename=rf"static/videos/output/output_{filename}"
    # processed_filename=rf"output_{filename}"
    out_file=rf"videos/output/output_{filename}"

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(video)
    video_writer = None
    # if args.out:
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
    video_writer = cv2.VideoWriter(
        processed_filename, fourcc, video_reader.fps,
        (video_reader.width, video_reader.height))

    for frame in track_iter_progress(video_reader):
        frame2 = frame.copy()
        lines = lane(frame2)
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=40
            )
        frame = visualizer.get_image()

        frame = plot_lines(frame,lines)
        video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    return out_file