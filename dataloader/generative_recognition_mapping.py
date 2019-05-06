import copy

class GR_Map(object):
    def __init__(self, tc):
        self.tc = tc
        self.generative_recognition_map = {}

    def get_gr_map(self):
        return copy.deepcopy(self.generative_recognition_map)

class GR_Map_full(GR_Map):
    def __init__(self, tc):
        super(GR_Map_full, self).__init__(tc)
        stn_ids = {
            'rotate': 0, 
            'scale': 1, 

            'translate_up_small': 2,
            'translate_down_small': 3,
            'translate_left_small': 4,
            'translate_right_small': 5,

            'translate_up_normal': 6,
            'translate_down_normal': 7,
            'translate_left_normal': 8,
            'translate_right_normal': 9,

            'translate_up_big': 10,
            'translate_down_big': 11,
            'translate_left_big': 12,
            'translate_right_big': 13,

            'identity': 14}

        # generative ids
        rotate_ids = [x.id for x in self.tc.rotate]
        scale_ids = [x.id for x in self.tc.scale]
        identity_ids = [self.tc.identity.id]

        for i in rotate_ids:
            self.generative_recognition_map[i] = stn_ids['rotate']
        for i in scale_ids:
            self.generative_recognition_map[i] = stn_ids['scale']

        for k, v in filter(lambda (k, v): 'translate' in k, stn_ids.iteritems()):
            self.generative_recognition_map[v+2] = stn_ids[k]

        for i in identity_ids:
            self.generative_recognition_map[i] = stn_ids['identity']