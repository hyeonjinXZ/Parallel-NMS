from parameters import get_parameter, view_parameter
from network_train import learn_detector_network

if __name__ == '__main__':
    # get parser
    args = get_parameter()
    view_parameter(args)
    
    # trainin detector network
    detector, loss_calculator, optimizer = learn_detector_network(args)
