class TargetValueMapping:

    def __init__(self):
        self.neg:int = 0
        self.pos:int = 1


    def to_dict(self) -> dict:
        """
        Convert the target value mapping to a dictionary.
        """
        return self.__dict__
    
    def reverse_mapping(self):
        """
        Reverse the target value mapping.
        """
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))