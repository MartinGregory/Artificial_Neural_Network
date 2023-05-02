//
//
//  MultiLayerPerceptron.hpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#ifndef _MultiLayerPerceptron_H_
#define _MultiLayerPerceptron_H_

#include "Perceptron.hpp"
//#include "Globals.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>       // rand(), srand()
#include <time.h>       // time(0) i.e. the current time that seeds srand()
#include <algorithm>    // needed for all STL algorithms


//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
class MultiLayerPerceptron {
    
// DATA MEMBERS (private by default)
    
    std::vector< int > layers ;     /* Each element represents the number of neurons per layer.
    Note that layers[0] represents the input layer (which has NO neurons) and the value represents the
    number of inputs. */
    
    double eta ;               /* by convention the lowercase Greek letter η eta (pronounce ay-tah)
                               is being used to demarcate the "learning rate" of the network.
    I.e. it impacts the rate at which the Gradient Descent is "descending" towards the local
    (preferably global) minimum */
    
    std::vector< std::vector< Perceptron >> network ;       /* network of layers of Neurons ; each
    element (of the outer vector) represents its corresponding layer of Neurons (Perceptrons);
    each element of the inner vector (layer) is an individual Neuron (Perceptron) */
    
    std::vector< std::vector< double >> output_values ;    /* here we'll store the output values from
    each of the neurons (each neuron produces one output value). We need these values so that we can
    propagate forward via the network. */
    
    std::vector< std::vector< double >> errors ;           /* here we'll store the error terms for each
    of the neurons. */
  
// MEMBER FUNCTIONS (public interface)
public:
    
    MultiLayerPerceptron( std::vector< int > layers, double eta=0.5 ) ;
    
    std::vector<double> run( std::vector< double > inputs ) ;
    
    double bp( std::vector< double > x, std::vector< double > y ) ;
    
    void set_weights( std::vector< std::vector< std::vector< double >>> init_vec ) ;
    void print_weights() const ;
};//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#endif /* MultiLayerPerceptron.hpp */
