//
//  Perceptron.cpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#include "Perceptron.hpp"
//#include <iostream>
//#include <vector>
//#include <numeric>
//#include <cmath>
//#include <random>       // rand(), srand()
//#include <time.h>       // time(0) i.e. the current time that seeds srand()
//#include <algorithm>    // needed for all STL algorithms


//....................................................................................................
/* This function will generate a different random number for every call. Since these values will be
 used as weights for the Perceptron model we want to constrain them to be within -1 and +1 */
double generate_random(){
    
    //srand( u_int(time(0)) ) ;   // srand() uses the current time as seed for random generator rand()
    // do the above in the main()
    
    double val = 0.0 ;
    val = double( rand() ) ;        // generate numbers 0 to RAND_MAX
    val /= RAND_MAX ;               // normalize the range within 0...1
    val *= 2.0 ;                    // scale/expand the range to be within 0...2
    val -= 1.0 ;                    // now shift the range within -1...1
    
    //val = (( 2.0 * (double)rand() / RAND_MAX ) - 1.0) ;   // OR a simple one liner
    return val ;
}
//....................................................................................................
//Return a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron( int input_size , double bias ): bias{bias}, input_size{input_size}{
    
    weights.resize( input_size+1 ) ;        // +1 to adjust for the bias
    
    // Populate the vector of weights with random values
    std::generate( weights.begin(), weights.end(), generate_random ) ;
    
    //std::for_each( weights.begin(), weights.end(), [](const double& v){ std::cout << v <<" ";}) ;
    //std::cout << std::endl ;
}
//....................................................................................................
//Run the Perceptron. The formal parameter 'input' is the vector containing the input values.
double Perceptron::run( std::vector< double > inputs ){
    
    inputs.emplace_back( (*this).bias ) ;
    
    // Weighted Sum is the Dot Product of the inputs and weights vectors
    double weighted_sum = std::inner_product( inputs.begin(), inputs.end(), (*this).weights.begin(), 0.0 );
    
    return weighted_sum ;
}
//....................................................................................................
/* The Weighted Sum (Dot Product) 'z' outputed from the Perceptron must now pass via a non-linear
activation function. One of such functions is a Sigmoid (Logistic) function: 1/(1+e^-z)*/
double Perceptron::sigmoid( double z ){
    return 1.0 / ( 1.0 + exp(-z) ) ;
}
//....................................................................................................
/* The 'w_inti' is a vector containing weights which we want to use to initialize the vector of
weights */
void Perceptron::set_weights( std::vector< double > init_vec ){
    
    if( init_vec.size()+1 != (*this).input_size+1 )
        init_vec.resize( (*this).input_size+1 ) ;
    
    (*this).weights = std::move( init_vec ) ;
}
//....................................................................................................
