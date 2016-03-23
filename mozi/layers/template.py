

class Template(object):
    """
    DESCRIPTION:
        The interface to be implemented by any layer.
    """
    def __init__(self):
        '''
        FIELDS:
            self.params: any params from the layer that needs to be updated
                         by backpropagation can be put inside self.params
            self.updates: use for updating any shared variables
        '''
        self.params = []
        self.updates = []

    def _test_fprop(self, state_below):
        '''
        DESCRIPTION:
            This is called during validating/testing of the model.
        PARAM:
            state_below: the input to layer
        '''
        raise NotImplementedError(str(type(self))+" does not implement _test_fprop.")


    def _train_fprop(self, state_below):
        '''
        DESCRIPTION:
            This is called during every training batch whereby the output from the
            model will be used to update the parameters during error backpropagation.
        PARAM:
            state_below: the input to layer
        '''
        raise NotImplementedError(str(type(self))+" does not implement _train_fprop.")


    def _layer_stats(self, state_below, layer_output):
        """
        DESCRIPTION:
            Layer stats is used for debugging the layer by allowing user to peek
            at the weight values or the layer output or any parameters of interest
            during training. By computing the values of parameter of interest,
            for example T.max(self.W) and put in the return list, the training will
            print the maximum of the weight in the layer after every epoch.
        PARAM:
            state_below: the input to layer
            layer_output: the output from the layer
        RETURN:
            A list of tuples of [('name_a', var_a), ('name_b', var_b)] whereby var is scalar
        """
        return []
