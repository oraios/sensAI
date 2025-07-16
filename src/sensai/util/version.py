

class Version:
    """
    Assists in checking the version of a Python package based on the __version__ attribute
    """
    def __init__(self, package):
        """
        :param package: the package object
        """
        self.components = package.__version__.split(".")

    def __str__(self):
        return ".".join(self.components)

    def is_at_least(self, *components: int):
        """
        Checks this version against the given version components.
        This version object must contain at least the respective number of components

        :param components: version components in order (i.e. major, minor, patch, etc.)
        :return: True if the version is at least the given version, False otherwise
        """
        for i, desired_min_version in enumerate(components):
            actual_version = int(self.components[i])
            if actual_version < desired_min_version:
                return False
            elif actual_version > desired_min_version:
                return True
        return True

    def is_at_most(self, *components: int):
        """
        Checks this version against the given version components.
        This version object must contain at least the respective number of components

        :param components: version components in order (i.e. major, minor, patch, etc.)
        :return: True if the version is at most the given version, False otherwise
        """
        for i, desired_max_version in enumerate(components):
            actual_version = int(self.components[i])
            if actual_version > desired_max_version:
                return False
            elif actual_version < desired_max_version:
                return True
        return True

    def is_equal(self, *components: int):
        """
        Checks this version against the given version components.
        This version object must contain at least the respective number of components

        :param components: version components in order (i.e. major, minor, patch, etc.)
        :return: True if the version is the given version, False otherwise
        """
        return self.components[:len(components)] == list(components)
