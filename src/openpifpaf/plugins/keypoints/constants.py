import os
import numpy as np

from Experiments.openpifpaf.src.openpifpaf.plugins.animalpose.constants import draw_skeletons

_CATEGORIES = ['argoverse']

#Name: Name of the keypoints
LANE_KEYPOINTS = [
    'head', #1
    'control-pt1', #2
    'tail', #3
]

#Skeleton: Defines the connections between the keypoints
#First number in the tuple is the beginning keypoint and the second one is the end
#For a lane segment, we have head connected to control-pt1, control-pt1 to tail

#For a segment with just 2 keypoint, we need only 2 segmets (head-control point-tail)
LANE_SKELETON = [
    (1,2), (2, 3)
]

#Sigmas: Size of the area to compute the object key point similarity
#Adding equal weight of 0.05 for now
LANE_SIGMAS = [0.05] * len(LANE_KEYPOINTS)

split, error = divmod(len(LANE_KEYPOINTS), 4)

#Weights: These are the weights to compute the overall score of an object
#Highest weights are assigned to most confident joints
LANE_SCORE_WEIGHTS = [5.0] * split + [3.0] * split + [1.0] * split + [0.5] * split + [0.1] * error

#Categories: Represents the categories of the keypoints, in this case, keypoints are only of lanes
LANE_CATEGORIES = ['lane']

#Pose: Not sure yet about how to populate
LANE_POSE = np.array([])

assert len(LANE_POSE) == len(LANE_KEYPOINTS) == len(LANE_SIGMAS) \
       == len(LANE_SCORE_WEIGHTS), "Dimension mismatch!"

def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    show.KeypointPainter.font_size = 0
    keypoint_painter = show.KeypointPainter()

    ann = Annotation(
        keypoints=LANE_KEYPOINTS, skeleton=LANE_SKELETON, score_weights=LANE_SCORE_WEIGHTS)
    ann.set(pose, np.array(LANE_SIGMAS) * scale)
    os.makedirs('all-images', exist_ok=True)
    draw_ann(ann, filename='all-images/predicted_lane_segments.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in LANE_SKELETON:
        print(LANE_SKELETON[j1-1], '-', LANE_KEYPOINTS[j2-1])

if __name__ == '__main__':
    print_associations()
    draw_skeletons(LANE_POSE)