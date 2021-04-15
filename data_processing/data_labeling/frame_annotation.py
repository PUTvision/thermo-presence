import copy
from typing import List
from typing import Tuple

from data_labeling.labeling_utils import xy_on_interpolated_image_to_raw_xy, xy_on_raw_image_to_xy_on_interpolated_image


class FrameAnnotation:
    def __init__(self):
        self.accepted = False  # whether frame was marked as annotated successfully
        self.discarded = False  # whether frame was marked as discarded (ignored)
        self.centre_points = []  # type: List[tuple]  # x, y
        self.rectangles = []  # type: List[Tuple[tuple, tuple]]  # (x_left, y_top), (x_right, y_bottom)

        self.raw_frame_data = None  # Not an annotation, but write it to the result file, just in case

    def as_dict(self):
        data_dict = copy.copy(self.__dict__)
        data_dict['centre_points'] = []
        data_dict['rectangles'] = []

        for i, point in enumerate(self.centre_points):
            data_dict['centre_points'].append(xy_on_interpolated_image_to_raw_xy(point))
        for i, rectangle in enumerate(self.rectangles):
            data_dict['rectangles'].append((xy_on_interpolated_image_to_raw_xy(rectangle[0]),
                                            xy_on_interpolated_image_to_raw_xy(rectangle[1])))
        return data_dict

    @classmethod
    def from_dict(cls, data_dict, do_not_scale_and_reverse=False):
        item = cls()
        item.__dict__.update(data_dict)
        for i, point in enumerate(item.centre_points):
            item.centre_points[i] = xy_on_raw_image_to_xy_on_interpolated_image(point, do_not_scale_and_reverse)
        for i, rectangle in enumerate(item.rectangles):
            item.rectangles[i] = (xy_on_raw_image_to_xy_on_interpolated_image(rectangle[0], do_not_scale_and_reverse),
                                  xy_on_raw_image_to_xy_on_interpolated_image(rectangle[1], do_not_scale_and_reverse))
        return item
