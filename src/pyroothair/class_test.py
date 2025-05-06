


class Root():
    def __init__(self, length:int):
        self.root_thickness = 10
        self.length = length   
        self.area = None
        self.make_area()

    def make_area(self):
        self.area = self.length ** 2
        print('Area', self.area)

class Params(Root):
    def __init__(self, length:int):
        super().__init__(length)
        print(self.root_thickness)

        print(length)
    
        print(self.area)


def main():
    x = Root(10)
    
    z = Params(10)

if __name__ == '__main__':
    main()
    