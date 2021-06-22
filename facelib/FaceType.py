from enum import IntEnum

class FaceType(IntEnum):
    #enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    FS_FACE = 5
    FFHQ = 6
    BIG_FACE = 7
    SMALL_HEAD = 8
    FS_HEAD = 9
    HEAD = 10
    HEAD_NO_ALIGN = 20

    MARK_ONLY = 100, #no align at all, just embedded faceinfo

    @staticmethod
    def names():
        return to_string_dict.values()

    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_dict[face_type]

    @staticmethod
    def abbreviations():
        return to_abbreviation_dict.values()

    @staticmethod
    def fromAbbreviation (s):
        r = from_abbreviation_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromAbbreviation value error')
        return r

    @staticmethod
    def toAbbreviation (face_type):
        return to_abbreviation_dict[face_type]

to_string_dict = { FaceType.HALF : 'half_face',
                   FaceType.MID_FULL : 'midfull_face',
                   FaceType.FULL : 'full_face',
                   FaceType.FULL_NO_ALIGN : 'full_face_no_align',
                   FaceType.WHOLE_FACE : 'whole_face',
                   FaceType.FS_FACE : 'fs_face',
                   FaceType.FFHQ : 'ffhq',
                   FaceType.BIG_FACE : 'big_face',
                   FaceType.SMALL_HEAD: 'small_head',
                   FaceType.FS_HEAD: 'fs_head',
                   FaceType.HEAD : 'head',
                   FaceType.HEAD_NO_ALIGN : 'head_no_align',
                   
                   FaceType.MARK_ONLY :'mark_only',  
                 }

from_string_dict = { to_string_dict[x] : x for x in to_string_dict.keys() }

to_abbreviation_dict = {FaceType.HALF: 'h',
                  FaceType.MID_FULL: 'mf',
                  FaceType.FULL: 'f',
                  FaceType.WHOLE_FACE: 'wf',
                  FaceType.FS_FACE : 'fsf',
                  FaceType.FFHQ : 'ffhq',
                  FaceType.BIG_FACE: 'bf',
                  FaceType.SMALL_HEAD: 'sh',
                  FaceType.FS_HEAD: 'fsh',
                  FaceType.HEAD: 'head',
                  }

from_abbreviation_dict = {to_abbreviation_dict[x]: x for x in to_abbreviation_dict.keys()}
