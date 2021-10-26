"""
VideoFrameGenerator - Simple Generator
--------------------------------------
A simple frame generator that takes distributed frames from
videos. It is useful for videos that are scaled from frame 0 to end
and that have no noise frames.
"""

import os
import glob
import numpy as np
import cv2 as cv
from math import floor
import logging
import re
log = logging.getLogger()

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator, img_to_array


class VideoFrameGenerator(Sequence):
    """
    Create a generator that return batches of frames from video
    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that \
        will be replaced by one of the class list
    - use_header: bool, default to True to use video header to read the \
        frame count if possible

    You may use the "classes" property to retrieve the class list afterward.
    The generator has that properties initialized:
    - classes_count: number of classes that the generator manages
    - files_count: number of video that the generator can provides
    - classes: the given class list
    - files: the full file list that the generator will use, this \
        is usefull if you want to remove some files that should not be \
        used by the generator.
    """

    def __init__(
            self,
            rescale=1/255.,
            nb_frames: int = 5,
            classes: list = None,
            batch_size: int = 16,
            use_frame_cache: bool = False,
            target_shape: tuple = (224, 224),
            shuffle: bool = True,
            transformation: ImageDataGenerator = None,
            split_test: float = None,
            split_val: float = None,
            nb_channel: int = 3,
            frame_step: int = 6,
            glob_pattern: str = './videos/{classname}/*.avi',
            use_headers: bool = True,
            *args,
            **kwargs):

        # deprecation
        if 'split' in kwargs:
            log.warn("Warning, `split` argument is replaced by `split_val`, "
                     "please condider to change your source code."
                     "The `split` argument will be removed "
                     "in future releases.")
            split_val = float(kwargs.get('split'))

        self.glob_pattern = glob_pattern

        # should be only RGB or Grayscale
        assert nb_channel in (1, 3)

        if classes is None:
            classes = self._discover_classes()

        # we should have classes
        if len(classes) == 0:
            log.warn("You didn't provide classes list or "
                     "we were not able to discover them from "
                     "your pattern.\n"
                     "Please check if the path is OK, and if the glob "
                     "pattern is correct.\n"
                     "See https://docs.python.org/3/library/glob.html")

        # shape size should be 2
        assert len(target_shape) == 2

        # split factor should be a propoer value
        if split_val is not None:
            assert 0.0 < split_val < 1.0

        if split_test is not None:
            assert 0.0 < split_test < 1.0

        self.use_video_header = use_headers

        # then we don't need None anymore
        split_val = split_val if split_val is not None else 0.0
        split_test = split_test if split_test is not None else 0.0

        # be sure that classes are well ordered
        classes.sort()

        self.rescale = rescale
        self.classes = classes
        self.batch_size = batch_size
        self.nb_frames = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation
        self.use_frame_cache = use_frame_cache
        self.frame_step = frame_step

        self._random_trans = []
        self.__frame_cache = {}
        self.files = []
        self.validation_files = []
        self.test_files = []

        _validation_data = kwargs.get('_validation_data', None)
        _test_data = kwargs.get('_test_data', None)

        if _validation_data is not None:
            # we only need to set files here
            self.files = _validation_data

        elif _test_data is not None:
            # we only need to set files here
            self.files = _test_data

        else:
            if split_val > 0 or split_test > 0:
                for cls in classes:
                    files = glob.glob(glob_pattern.format(classname=cls))
                    nbval = 0
                    nbtest = 0
                    info = []

                    # generate validation and test indexes
                    file_indexes = np.arange(len(files))

                    if shuffle:
                        np.random.shuffle(file_indexes)

                    if 0.0 < split_val < 1.0:
                        nbval = int(split_val * len(files))
                        nbtrain = len(files) - nbval

                        # get some sample for validation_data
                        val = np.random.permutation(file_indexes)[:nbval]

                        # remove validation from train
                        file_indexes = np.array(
                            [i for i in file_indexes if i not in val])
                        self.validation_files += [files[i] for i in val]
                        info.append("validation count: %d" % nbval)

                    if 0.0 < split_test < 1.0:
                        nbtest = int(split_test * nbtrain)
                        nbtrain = len(files) - nbval - nbtest

                        # get some sample for test_data
                        val_test = np.random.permutation(file_indexes)[:nbtest]

                        # remove test from train
                        file_indexes = np.array(
                            [i for i in file_indexes if i not in val_test])
                        self.test_files += [files[i] for i in val_test]
                        info.append("test count: %d" % nbtest)

                    # and now, make the file list
                    self.files += [files[i] for i in file_indexes]
                    print("class %s, %s, train count: %d" %
                          (cls, ", ".join(info), nbtrain))

            else:
                for cls in classes:
                    self.files += glob.glob(glob_pattern.format(classname=cls))

        # build indexes
        self.files_count = len(self.files)
        self.file_indexes = np.arange(self.files_count)
        self.classes_count = len(classes)

        print("counting frames...")
        self.num_frames = 0
        self.num_items = 0
        self._framecounters = {}
        self._items_per_file = {} # file -> [(i, video_name, first_frame_position, last_frame_position)]
        for i in range(len(self.files)):
            video = self.files[i]
            cap = cv.VideoCapture(video)
            self.num_frames += self.count_frames(cap, i)
            cap.release()

        if self.shuffle:
            np.random.shuffle(self.file_indexes)
        self._items = self.create_items_list(self.file_indexes)
   
        #print("num_frames=", self.num_frames, "num_sequences=", self.num_sequences)
        #for i in self._items:
        #     print(i)

        # to initialize transformations and shuffle indices
        if 'no_epoch_at_init' not in kwargs:
            self.on_epoch_end()

        kind = "train"
        if _validation_data is not None:
            kind = "validation"
        elif _test_data is not None:
            kind = "test"

        self._current = 0
        print("Total data: %d classes for %d files for %s" % (
            self.classes_count,
            self.files_count,
            kind))

    def create_items_list(self, file_indexes):
        items = []
        #print("create_items_list")
        for file_index in file_indexes:
            #print("file_index", file_index)
            items += self._items_per_file[file_index]
        return items

    def count_frames(self, cap, file_index, force_no_headers=False):
        """ Count number of frame for video
        if it's not possible with headers """
        file_name = self.files[file_index]
        if not force_no_headers and file_name in self._framecounters:
            return self._framecounters[file_name]

        total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)

        if force_no_headers or total_frames < 0:
            # headers not ok
            total = 0
            # TODO: we're unable to use CAP_PROP_POS_FRAME here
            # so we open a new capture to not change the
            # pointer position of "cap"
            c = cv.VideoCapture(file_name)
            while True:
                grabbed, frame = c.read()
                if not grabbed:
                    # rewind and stop
                    break
                total_frames += 1

        # keep the result
        self._framecounters[file_name] = int(total_frames)

        chunk_offset = (self.frame_step - 1) * self.nb_frames
        start_frame_num = 0
        end_frame_num = chunk_offset
        self._items_per_file[file_index] = []
        while end_frame_num < int(total_frames):
            self._items_per_file[file_index].append((self.num_items, file_name, start_frame_num, end_frame_num))
            self.num_items += 1
            start_frame_num += 1
            end_frame_num += 1
            if start_frame_num % self.nb_frames == 0:
                # move to the next chunk in the video
                start_frame_num += chunk_offset
                end_frame_num += chunk_offset

        return total_frames

    def _discover_classes(self):
        pattern = os.path.realpath(self.glob_pattern)
        pattern = re.escape(pattern)
        pattern = pattern.replace('\\{classname\\}', '(.*?)')
        pattern = pattern.replace('\\*', '.*')

        files = glob.glob(self.glob_pattern.replace('{classname}', '*'))
        classes = set()
        for f in files:
            f = os.path.realpath(f)
            cl = re.findall(pattern, f)[0]
            classes.add(cl)

        return list(classes)

    def next(self):
        """ Return next element"""
        elem = self[self._current]
        self._current += 1
        if self._current == len(self):
            self._current = 0
            self.on_epoch_end()

        return elem

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            nb_frames=self.nb_frames,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            use_headers=self.use_video_header,
            _validation_data=self.validation_files)

    def get_test_generator(self):
        """ Return the test generator if you've provided split factor """
        return self.__class__(
            nb_frames=self.nb_frames,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            use_headers=self.use_video_header,
            _test_data=self.test_files)

    def on_epoch_end(self):
        """ Called by Keras after each epoch """

        if self.transformation is not None:
            self._random_trans = []
            for _ in range(self.num_items):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        self.__frame_cache = {} # clear the old frame cache

        if self.shuffle:
            print("\non_epoch_end: shuffle\n")
            np.random.shuffle(self.file_indexes)
            self._items = self.create_items_list(self.file_indexes)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.num_items // self.batch_size

    def __getitem__(self, index):
        classes = self.classes
        shape = self.target_shape
        nb_frames = self.nb_frames

        labels = []
        images = []

        batch_items = self._items[index*self.batch_size:(index+1)*self.batch_size]

        #print(F"getitem index={index} batch_size={self.batch_size} batch_items={batch_items}")

        transformation = None

        for i, video, start_frame, end_frame in batch_items:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            #video = self.files[i]
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            if video not in self.__frame_cache:
                self.__frame_cache = {} # clear the old frame cache

                frames = self._get_frames(video)
                if frames is None:
                    # failed to get frames
                    print("failed to read video", video)
                    exit(-1)

                # apply transformation
                if transformation is not None:
                    frames = [self.transformation.apply_transform(
                        frame, transformation) for frame in frames]

                # add to cache
                self.__frame_cache[video] = frames

            else:
                frames = self.__frame_cache[video]

            # append the selected frames
            selected_frames = []
            frame = start_frame
            while frame <= end_frame:
                selected_frames.append(frames[frame])
                frame += self.frame_step

            # add the sequence in batch
            images.append(selected_frames)
            labels.append(label)

        return np.array(images), np.array(labels)

    def _get_classname(self, video: str) -> str:
        """ Find classname from video filename following the pattern """

        # work with real path
        video = os.path.realpath(video)
        pattern = os.path.realpath(self.glob_pattern)

        # remove special regexp chars
        pattern = re.escape(pattern)

        # get back "*" to make it ".*" in regexp
        pattern = pattern.replace('\\*', '.*')

        # use {classname} as a capture
        pattern = pattern.replace('\\{classname\\}', '(.*?)')

        # and find all occurence
        classname = re.findall(pattern, video)[0]
        return classname

    def _get_frames(self, video):
        #print("get frames from", video, "\n")
        cap = cv.VideoCapture(video)
        expected_frames = self._framecounters[video]
        frames = []

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            # resize
            frame = cv.resize(frame, self.target_shape)

            # use RGB or Grayscale ?
            if self.nb_channel == 3:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            # to np
            frame = img_to_array(frame) * self.rescale

            frames.append(frame)

        cap.release()

        if expected_frames != len(frames):
            print("bad frame count: ", len(frames), "vs", expected_frames)
            exit(-1)

        return np.array(frames)
