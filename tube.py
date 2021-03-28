class Tube:
    def __init__(self, val=None, child=None):
        self.child = child
        self.val = val

    @staticmethod
    def get_tube(arr):
        assert len(arr) != 0
        head = Tube(arr[0])
        pointer = head
        for el in arr[1:]:
            pointer.child = Tube(el)
            pointer = pointer.child
        return head

    def __getitem__(self, item):
        if item == 0:
            return self
        else:
            if self.child is not None:
                return self.child.__getitem__(item-1)
            else:
                raise ValueError("incorrect index")

    def pop(self):
        if self.child is None:
            return
        else:
            if self.child.child is None:
                self.child = None
            else:
                self.child.pop()
        return self

    def pushed(self, val):
        new_head = Tube(val)
        new_head.child = self
        return new_head

    def tolist(self):
        result = list()
        point = self
        while point is not None:
            result.append(point.val)
            point = point.child
        return result

    def __str__(self):
        point = self
        if point.child is None:
            return str(point.val)
        result = ""
        while point.child is not None:
            result += "{}->".format(point.val)
            point = point.child
        result += str(point.val)
        return result

    def mean(self):
        point = self
        num = 0
        sum = 0
        while point is not None:
            num += 1
            sum += point.val
            point = point.child
        if num:
            return sum/num
        else:
            return 0