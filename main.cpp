//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#include "Perceptron.hpp"
#include <iostream>

//....................................................................................................
/* In this test we are running the Perceptron model to simulate Logic AND Gate. There are only two
dimensions X and Y which represent points (0,0), (0,1), (1,0) and (1,1). These points are linearly
separable and are on a 2D Cartesian Plane. These points represent the values of the AND Table:
    0 and 0 = 0     input( 0,0 )
    0 and 1 = 0     input( 0,1 )
    1 and 0 = 0     input( 1,0 )
    1 and 1 = 1     input( 1,1 )
The task for the Perceptron model is to find the decision boundry that correctly seperates/classifies
the input are either 0 (false) or 1 (true). Note that since this data is linearly separable all we
need is a single Perceptron -- hence our Neural Network will only have one node.
*/
void test0(){
    
    Perceptron * p {nullptr} ;
    p = new Perceptron( 2 ) ;   // for Logic AND gate input size is 2
    if( p == nullptr )
        return ;
    
    (*p).set_weights( {10,10,-15} ) ;   // these weights are NOT learned by the network (just testing)
    
    std::cout << "AND GATE:\n" ;
    std::cout <<"0 AND 0 : "<< (((*p).sigmoid( (*p).run( {0,0} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"0 AND 1 : "<< (((*p).sigmoid( (*p).run( {0,1} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 AND 0 : "<< (((*p).sigmoid( (*p).run( {1,0} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 AND 1 : "<< (((*p).sigmoid( (*p).run( {1,1} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    
    delete p ;
}
//....................................................................................................
//....................................................................................................
/* In this test we are running the Perceptron model to simulate Logic OR Gate. There are only two
dimensions X and Y which represent points (0,0), (0,1), (1,0) and (1,1). These points are linearly
separable and are on a 2D Cartesian Plane. These points represent the values of the OR Table:
    0 and 0 = 0     input( 0,0 )
    0 and 1 = 1     input( 0,1 )
    1 and 0 = 1     input( 1,0 )
    1 and 1 = 1     input( 1,1 )
The task for the Perceptron model is to find the decision boundry that correctly seperates/classifies
the input are either 0 (false) or 1 (true). Note that since this data is linearly separable all we
need is a single Perceptron -- hence our Neural Network will only have one node.
*/
void test1(){
    
    Perceptron * p  {nullptr} ;
    p = new Perceptron( 2 ) ;   // for Logic OR gate input size is 2
    if( p == nullptr )
        return ;
    
    (*p).set_weights( {15,15,-10} ) ;   // these weights are NOT learned by the network (just testing)
    
    std::cout << "OR GATE:\n" ;
    std::cout <<"0 OR 0 : "<< (((*p).sigmoid( (*p).run( {0,0} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"0 OR 1 : "<< (((*p).sigmoid( (*p).run( {0,1} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 OR 0 : "<< (((*p).sigmoid( (*p).run( {1,0} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 OR 1 : "<< (((*p).sigmoid( (*p).run( {1,1} ))) > 0.5 ? 1 : 0 ) << std::endl ;
    
    delete p ;
}
//....................................................................................................

//....................................................................................................
int main() {
    
    srand( u_int( time(NULL) ) ) ; // srand() uses the current time as seed for random generator rand()
    rand() ;
    
    test0() ;
    test1() ;
    
    return 0;
}//...................................................................................................
