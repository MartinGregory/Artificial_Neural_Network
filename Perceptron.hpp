//
//
//  Perceptron.hpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#ifndef _Perceptron_H_
#define _Perceptron_H_

//#include "Globals.hpp"
//#include "MultiLayerPerceptron.hpp"

#include <iostream>
#include <vector>
#include <numeric>          // std::inner_product()
#include <cmath>
#include <random>           // rand(), srand()
#include <time.h>           // time(0) i.e. the current time that seeds srand()
#include <algorithm>        // needed for all STL algorithms

enum class ACTIVATION{ Sigmoid, TanH, ReLu } ;

//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
class Perceptron {
    
    friend class MultiLayerPerceptron ;

    /* Input Layer i.e. the vector of features describing the input data. Note that input.size()
    will give you the number of features in the input vector excluding bias. */
    std::vector< double > input ;

    /* ALL values must be Real values (NO Integers) -- vector of weights --i.e. the vector containing
    values that the Neuron must learn on its own via backpropagation */
    std::vector< double > weights ;

    /* The 'activation_flag' determines which activation function to apply to the Dot Product. It
    will also signal the backpropagation algorithm which activation function was used (by the given
    neuron) so that the proper derivative can be computed */
    ACTIVATION activation_flag ;

    /* Output Layer: Neuron’s output after passing via activation function. Note that each neuron
    outputs ONLY a single output value */
    double output ;

public:
    Perceptron( int input_size , ACTIVATION flag=ACTIVATION::Sigmoid ) ;
    
    double run( std::vector< double > features ) ;
    
    void set_weights( std::vector< double > init_vec ) ;
    std::vector<double> & get_weights() ;
    void print_weights() ;
};//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#endif /* Perceptron_hpp */
