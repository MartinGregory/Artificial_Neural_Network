//
//
//  MultiLayerPerceptron.hpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#ifndef _MultiLayerPerceptron_H_
#define _MultiLayerPerceptron_H_

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>       // rand(), srand()
#include <time.h>       // time(0) i.e. the current time that seeds srand()
#include <algorithm>    // needed for all STL algorithms

//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
class MultiLayerPerceptron {
    
    std::vector< int > layers ;     /* Each element represents the number of neurons per layer.
    Note that this also includes the input layer (which has NO neurons) but here we mean the number
    of inputs. */
    
    double bias ;
    deouble eta ;                   // learning rate
    
    std::vector< std::vector< Perceptron >> network ;
    std::vector< std::vector< double >> out_values ;    // here we'll store the output values from
    // the neurons. We need these values so that we can propagate forward via the network.
    
    std::vector< std::vector< double >> errors ;        // here we'll store the error terms for
    // each of the neurons.
    
public:
    
    MultiLayerPerceptron( std::vector< int > layers, double bias=1.0, double eta=0.5 ) ;
    
    std::vector<double> run( std::vector< double > inputs ) ;
    void set_weights( std::vector< std::vector< std::vector< double >>> init_vec ) ;
    void print_weights() ;
    double bp( std::vector< double > x, std::vector< double > y ) ;
    
    double sigmoid( double z ) ;
};//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#endif /* MultiLayerPerceptron.hpp */
