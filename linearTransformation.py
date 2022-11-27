from operator import itemgetter


class LinearTransformation:
    def __init__(self, p1, p2) -> None:
        p1, p2 = sorted((p1, p2), key=itemgetter(0))
        self.a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        self.b = p1[1] - self.a * p1[0]

    def transform(self, x):
        return self.a * x + self.b
