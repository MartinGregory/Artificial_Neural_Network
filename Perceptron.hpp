//
//
//  Perceptron.hpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#ifndef _Perceptron_H_
#define _Perceptron_H_

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>       // rand(), srand()
#include <time.h>       // time(0) i.e. the current time that seeds srand()
#include <algorithm>    // needed for all STL algorithms

//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
class Perceptron {
    
    // ALL values must be Real values (NO Integers)
    std::vector< double > weights ;     // vector of weights--to be trained
    double bias ;                       // bias allows for the decision boundry to be independent of
                                        // the origin
    int input_size ;
    
public:
    
    Perceptron( int input_size, double bias=1.0 ) ;
    
    double run( std::vector< double > inputs ) ;
    void set_weights( std::vector< double > init_vec ) ;
    double sigmoid( double z ) ;
};//||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#endif /* Perceptron_hpp */
