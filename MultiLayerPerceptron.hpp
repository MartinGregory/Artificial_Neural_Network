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
#include "Perceptron.hpp"

//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
class MultiLayerPerceptron {
    
// DATA MEMBERS (private by default)
       
    /* Start by defining the 'shape' of the network -- i.e.:
    - shape[0] : how many inputs in the feature vector (input layer has NO neurons)
    - shape[1] : how many neurons in the 1st layer
    - shape[2] : how many neurons in the 2nd layer
    - shape[n] : how many neurons in the nth layer, where n>0 */
    std::vector< int > shape ;
    
    /* By convention the lowercase Greek letter η eta is being used to demarcate the "learning rate
    of the network. I.e. it impacts the rate at which the Gradient Descent is "descending" towards
    the local (preferably global) minimum */
    double eta ;
    
    /* The Network of layers of Neurons ; the outer vector represents the layers i.e. each element
    corresponds to a layer of Neurons (Perceptrons); the inner vector contains individual Neurons
    belonging to the given layer */
    std::vector< std::vector< Perceptron >> network ;
    
    /* Here we'll store the output values from each of the neurons (each neuron produces one output
    value). We need these values so that we can propagate forward via the network. The output from
    ALL the preceding neurons is a combined input into each of the proceeding neurons.
    Warning! This vector does NOT contain the inputs of the input layer, it start from the 1st hidden
    layer*/
    std::vector< std::vector< double >> output_values ;
    
    /* Here we'll store the δ error terms for each of the neurons. */
    std::vector< std::vector< double >> errors ;
  
// MEMBER FUNCTIONS (public interface)
public:
    
    MultiLayerPerceptron( std::vector< int > shape, double eta=0.5, ACTIVATION flag=ACTIVATION::Sigmoid ) ;
    
    std::vector<double> run( std::vector< double > inputs ) ;
    
    double back_prop( std::vector< double > x , std::vector< double > y ) ;
    
    void set_weights( std::vector< std::vector< std::vector< double >>> init_vec ) ;
    void print_weights() const ;
};//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#endif /* MultiLayerPerceptron.hpp */
