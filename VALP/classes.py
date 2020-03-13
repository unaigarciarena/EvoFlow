from functools import reduce


class Connection(object):
    def __init__(self, input_component, output_component, info, name):
        """
        :param input_component: Input of the connection (the component providing the data)
        :param output_component: Output of the connection (the component receiving the data)
        Both parameters need to be indices
        """
        self.input = input_component
        self.info = info
        self.output = output_component
        self.name = name

    def print(self):
        return self.name + ", from " + self.input + " to " + self.output + ", " + self.info.print()

    def codify(self):
        return self.name + "_" + self.input + "_" + self.output + "_" + self.info.type + "_" + str(self.info.size)


class InOut(object):
    def __init__(self, data_type, size):
        """
        :param data_type: Type of the data
        :param size: Size of the data
        :return:
        """
        self.type = data_type
        self.size = size

    def print(self):
        return "Type: " + self.type + ", Size: " + str(self.size)


class Component(object):
    """
    This class comprises all the nodes in the directed acyclic graph representation, i.e., V. Internal nodes, and source and
    sink nodes have their own classes, inheriting from this one.
    """
    def __init__(self, taking, producing, depth, kind):

        if (type(taking) is not InOut and taking is not None) or (type(producing) is not InOut and producing is not None):
            raise Exception("Both the input and output of the model components must be InOut Objects")

        self.taking = taking
        self.producing = producing
        self.depth = depth
        self.type = kind

    def print(self):
        string = self.type + "\n"
        if self.taking is not None:
            string += "Taking:\n\t" + self.taking.print()
        else:
            string = ""
        if self.producing is not None:
            string += "\n\t Producing: \n\t" + self.producing.print()
        return string


class NetworkComp(Component):
    """
    N in the VALP notation
    """
    def __init__(self, descriptor, taking, producing, depth=None):
        """
        :param descriptor: Parameters of the network, NetworkDescriptor object
        """
        self.descriptor = descriptor  # Network descriptor
        super().__init__(taking, producing, depth, "Network")

    def change_input(self, inp):
        self.taking = inp
        self.descriptor.in_dim = inp
        self.descriptor.network.input_dim = max(inp, self.descriptor.network.input_dim)

    def update_output(self, inp):
        # print(inp, self.producing.size)
        if isinstance(inp, tuple):
            inp = reduce(lambda x, y: x*y, inp)

        self.producing.size = max(inp, self.producing.size)
        self.descriptor.output_dim = max(inp, self.descriptor.output_dim)

    def update_input(self, inp):
        self.taking.size = max(inp, self.taking.size)
        self.descriptor.input_dim = max(inp, self.descriptor.input_dim)

    def increase_input(self, size):
        self.taking.size += size
        self.descriptor.input_dim += size

    def increase_output(self, size):
        self.producing += size
        self.descriptor.output_dim += size


class ModelComponent(Component):
    """
    I U O in the VALP notation
    """
    def __init__(self, taking, producing, depth):
        super().__init__(taking, producing, depth, "Model")
