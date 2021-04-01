import json

from data_labeling.frame_annotation import FrameAnnotation


class AnnotationCollector:
    ANNOTATIONS_BETWEEN_AUTOSAVE = 10

    def __init__(self, output_file_path, data_batch_dir_path):
        self._output_file_path = output_file_path
        self._annotations = {}  # ir_frame_index: FrameAnnotation
        self._data_batch_dir_path = data_batch_dir_path
        self._annotations_to_autosave = 1

    def get_annotation(self, ir_frame_index):
        return self._annotations.get(ir_frame_index, FrameAnnotation())

    def set_annotation(self, ir_frame_index, annotation):
        self._annotations[ir_frame_index] = annotation
        if annotation.accepted or annotation.discarded:
            self._annotations_to_autosave -= 1
            if self._annotations_to_autosave == 0:
                self._annotations_to_autosave = self.ANNOTATIONS_BETWEEN_AUTOSAVE
                self.save()

    def save(self):
        data_dict = {
            'output_file_path': self._output_file_path,
            'data_batch_dir_path': self._data_batch_dir_path,
            'annotations': {index: annotation.as_dict()
                            for index, annotation in self._annotations.items()}
        }
        with open(self._output_file_path, 'w') as file:
            file.write(json.dumps(data_dict, indent=2))

    @classmethod
    def load_from_file(cls, file_path):
        item = cls(output_file_path=file_path, data_batch_dir_path=None)
        with open(file_path, 'r') as file:
            data = file.read()
        data_dict = json.loads(data)
        item._data_batch_dir_path = data_dict['data_batch_dir_path']
        item._annotations = {int(index): FrameAnnotation.from_dict(annotation_dict) for index, annotation_dict
                             in data_dict['annotations'].items()}
        return item
