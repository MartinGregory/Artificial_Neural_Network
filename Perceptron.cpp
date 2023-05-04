//
//  Perceptron.cpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#include "Globals.hpp"
#include "Perceptron.hpp"

extern const double bias ;  // bias allows for the decision boundry to be independent of
                            // the origin

//....................................................................................................
/* This function will generate a different random number for every call. Since these values will be
 used as weights for the Perceptron model we want to constrain them to be within -1 and +1 */
double generate_random(){
    
    // srand( u_int(time(0)) ) ;   // srand() uses the current time as seed for random generator rand()
    // do the above in the main()
    
    double val = 0.0 ;
    val = double( rand() ) ;        // generate numbers 0 to RAND_MAX
    val /= RAND_MAX ;               // normalize the range within 0...1
    val *= 2.0 ;                    // scale/expand the range to be within 0...2
    val -= 1.0 ;                    // now shift the range within -1...1
    
    // OR a simple one liner
    //val = (( 2.0 * (double)rand() / RAND_MAX ) - 1.0) ;
    return val ;
}
//....................................................................................................
//Returns a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron( int input_size ): input_size{input_size}  {
    
    // Save on memory space by resizing() the vector of weights to only accomodate the actual values
    weights.resize( input_size+1 ) ;        // +1 to adjust for the bias weight
    // Bias is always 1 but the bias weights must be learned by the network
    
    /* At first populate the vector of weights with random values. Later the actual values will be
    learned by the network */
    std::generate( weights.begin(), weights.end(), generate_random ) ;
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
void Perceptron::print_weights(){
    std::cout << "[ " ;
    std::for_each( weights.begin(), weights.end(), [](const double& w){ std::cout << w <<" ";} ) ;
    std::cout << "]\n" ;
}
//....................................................................................................
/* Returns the sequence of weights learned by the given Perceptron. */
std::vector<double> & Perceptron::get_weights() { return (*this).weights ; } ;
//....................................................................................................
//Run the Perceptron. The formal parameter 'input' is the vector containing the input values.
double Perceptron::run( std::vector< double > inputs , FUNC f ){
    
    inputs.emplace_back( bias ) ;
    
    (*this).input = inputs ;
    
    // Weighted Sum is the Dot Product of the inputs and weights vectors
    double weighted_sum = std::inner_product( inputs.begin(), inputs.end(), (*this).weights.begin(), 0.0 );

    (*this).output = f( weighted_sum ) ;
    return (*this).output ;
}
//....................................................................................................


