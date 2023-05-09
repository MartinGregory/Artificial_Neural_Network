//
//  Perceptron.cpp
//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#include "Globals.hpp"
#include "Perceptron.hpp"
#include "Activation_Function.cpp"

extern const double bias ;  // bias allows for the decision boundary to be independent of
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
Perceptron::Perceptron( int input_size, ACTIVATION flag ): activation_flag{flag} {
    
    /* Save on memory space by resizing() the vectors of inputs and weights to only accomodate the
    actual values. Note that I'm adding +1 to adjust for the bias. Also note that the 'bias' is always
    1 but the 'bias weights' must be learned by the network */
    input.resize( input_size+1 ) ;
    weights.resize( input_size+1 ) ;
    
    /* At first populate the vector of weights with random values. Later the actual values will be
    learned by the network */
    std::generate( weights.begin(), weights.end(), generate_random ) ;
}
//....................................................................................................
/* The 'w_inti' is a vector containing weights which we want to use to initialize the vector of
weights */
void Perceptron::set_weights( std::vector< double > init_vec ){
    
    if( init_vec.size()+1 != (*this).input.size()+1 )
        init_vec.resize( (*this).input.size()+1 ) ;
    
    (*this).weights = std::move( init_vec ) ;
}
//....................................................................................................
/* Returns the sequence of weights learned by the given Perceptron. */
std::vector<double> & Perceptron::get_weights() { return (*this).weights ; } ;
//....................................................................................................
void Perceptron::print_weights(){
    std::cout << "[ " ;
    std::for_each( weights.begin(), weights.end(), [](const double& w){ std::cout << w <<" ";} ) ;
    std::cout << "]\n" ;
}
//....................................................................................................
//Run the Perceptron. The formal parameter 'input' is the vector containing the input values.
double Perceptron::run( std::vector< double > features ){
    
    (*this).input = features ;
    (*this).input.emplace_back( bias ) ;
    
    // Weighted Sum is the Dot Product of the inputs and weights vectors
    double weighted_sum = std::inner_product( input.begin(), input.end(), (*this).weights.begin(), 0.0 );
    
    // pass the Dot Product via the given activation function
    Activation_Function f ;
    switch ( (*this).activation_flag ){
            
        case ACTIVATION::Sigmoid :
            (*this).output = f.sigmoid( weighted_sum ) ; break ;
            
        case ACTIVATION::TanH :
            (*this).output = f.tanh( weighted_sum ) ; break ;
            
        case ACTIVATION::ReLu :
            (*this).output = f.relu( weighted_sum ) ; break ;
            
        default:
            std::cerr << "Activation Function Error!\n" ;
    }
    return (*this).output ;
}
//....................................................................................................


