//  Neural_Network_FeedForward
//
//  Created by Martin Gregory Sendrowicz on 4/26/23.
//

#include "Perceptron.hpp"
#include "MultiLayerPerceptron.hpp"
#include "Globals.hpp"
#include "Activation_Function.cpp"
#include <iostream>
#include <memory>

//....................................................................................................
/* In this test we are running the Perceptron model to simulate Logic AND Gate. There are only two
dimensions X and Y which represent points (0,0), (0,1), (1,0) and (1,1). These points are linearly
separable and are on a 2D Cartesian Plane. These points represent the values of the AND Table:
    0 and 0 = 0     input( 0,0 )
    0 and 1 = 0     input( 0,1 )
    1 and 0 = 0     input( 1,0 )
    1 and 1 = 1     input( 1,1 )
The task for the Perceptron model is to find the decision boundry that correctly seperates/classifies
the given inputs as either 0 (false) or 1 (true). Note that since this data is linearly separable all
we need is a single Perceptron -- hence our Neural Network will only have one node. */
void test0(){
    
    Perceptron p = Perceptron( 2,ACTIVATION::Sigmoid ) ;    // Stack allocation
    
    //p.set_weights( {10,10,-15} ) ;   // these weights are NOT learned by the network (just testing)
    p.set_weights( {1,1,-1} ) ;   // these weights are NOT learned by the network (just testing)
    
    std::cout << "AND GATE:\n" ;
    std::cout <<"0 AND 0 : "<< (( p.run( {0,0} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"0 AND 1 : "<< (( p.run( {0,1} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 AND 0 : "<< (( p.run( {1,0} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 AND 1 : "<< (( p.run( {1,1} )) > 0.5 ? 1 : 0 ) << std::endl ;
}
//....................................................................................................
/* Below we simulate the AND Gate (just as above) using 2 inputs and 1 Perceptron--but this time
the network must learn its own weights via backpropagation. Note that since the output of the AND
Gate is linearly separable we can nust ReLU as our activation function.*/
void test0_1(){
    
    //MultiLayerPerceptron net = MultiLayerPerceptron( {2,1} ) ;
    //MultiLayerPerceptron net = MultiLayerPerceptron( {2,1},0.5,ACTIVATION::TanH ) ;
    MultiLayerPerceptron net = MultiLayerPerceptron( {2,1},0.5,ACTIVATION::ReLu ) ;
    
    double MSE = 0.0 ;
    int epochs = 100 ;
    
    for( int epoch=0 ; epoch < epochs ; ++epoch ){
            
        MSE += net.back_prop( {0,0},{0} ) ;
        MSE += net.back_prop( {0,1},{0} ) ;
        MSE += net.back_prop( {1,0},{0} ) ;
        MSE += net.back_prop( {1,1},{1} ) ;
        
        MSE /= 4.0 ;
        
        if( epoch % 10 == 0 )
            std::cout <<"Epoch: "<<epoch<<" MSE: "<<MSE<< std::endl ;
    }
    std::cout << "\nBelow are the trained/learned weights:" ;
    net.print_weights() ;
    
    std::cout << "AND GATE:\n" ;
    std::cout <<"0 AND 0 : "<< (((net.run( {0,0} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"0 AND 1 : "<< (((net.run( {0,1} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"1 AND 0 : "<< (((net.run( {1,0} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"1 AND 1 : "<< (((net.run( {1,1} )[0]) > 0.5) ? 1 : 0) << std::endl ;
}
//....................................................................................................
/* In this test we are running the Perceptron model to simulate Logic OR Gate. There are only two
dimensions X and Y which represent points (0,0), (0,1), (1,0) and (1,1). These points are linearly
separable and are on a 2D Cartesian Plane. These points represent the values of the OR Table:
    0 and 0 = 0     input( 0,0 )
    0 and 1 = 1     input( 0,1 )
    1 and 0 = 1     input( 1,0 )
    1 and 1 = 1     input( 1,1 )
The task for the Perceptron model is to find the decision boundry that correctly seperates/classifies
the given inputs as either 0 (false) or 1 (true). Note that since this data is linearly separable all
we need is a single Perceptron -- hence our Neural Network will only have one node. */
void test1(){
    
    Perceptron * p  {nullptr} ;                         // Heap allocation via raw pointer
    p = new Perceptron( 2,ACTIVATION::Sigmoid ) ;         // for Logic OR gate input size is 2
    if( p == nullptr )  // in case 'new' fails
        return ;
    
    (*p).set_weights( {15,15,-10} ) ;   // these weights are NOT learned by the network (just testing)
    
    std::cout << "OR GATE:\n" ;
    std::cout <<"0 OR 0 : "<< (( (*p).run( {0,0} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"0 OR 1 : "<< (( (*p).run( {0,1} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 OR 0 : "<< (( (*p).run( {1,0} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 OR 1 : "<< (( (*p).run( {1,1} )) > 0.5 ? 1 : 0 ) << std::endl ;
    
    delete p ;
}
//....................................................................................................
/* In this test we are running the Perceptron model to simulate Logic NAND Gate. There are only two
dimensions X and Y which represent points (0,0), (0,1), (1,0) and (1,1). These points are linearly
separable and are on a 2D Cartesian Plane. These points represent the values of the NAND Table:
    0 and 0 = 1     input( 0,0 )
    0 and 1 = 1     input( 0,1 )
    1 and 0 = 1     input( 1,0 )
    1 and 1 = 0     input( 1,1 )
The task for the Perceptron model is to find the decision boundry that correctly seperates/classifies
the given inputs as either 0 (false) or 1 (true). Note that since this data is linearly separable all
we need is a single Perceptron -- hence our Neural Network will only have one node. */
void test2(){
    
    // Heap allocation via smart pointer
    std::unique_ptr< Perceptron > p { new Perceptron( 2,ACTIVATION::Sigmoid ) } ;
    
    (*p).set_weights( {-10,-10,15} ) ;   // these weights are NOT learned by the network (just testing)
    
    std::cout << "NAND GATE:\n" ;
    std::cout <<"0 NAND 0 : "<< (( (*p).run( {0,0} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"0 NAND 1 : "<< (( (*p).run( {0,1} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 NAND 0 : "<< (( (*p).run( {1,0} )) > 0.5 ? 1 : 0 ) << std::endl ;
    std::cout <<"1 NAND 1 : "<< (( (*p).run( {1,1} )) > 0.5 ? 1 : 0 ) << std::endl ;

}
//....................................................................................................
/* In this test we are running a FeedForward Neural Network, composed of 3 Perceptrons to simulate
Logic XOR Gate. The input has two dimensions X and Y which represent points (0,0), (0,1), (1,0) and
(1,1). These points are NOT linearly separable on a 2D Cartesian Plane -- thus a single Perceptron
model will NOT be able to simulate the logic of the XOR Gate. Below is the XOR Table:
     0 and 0 = 0     input( 0,0 )
     0 and 1 = 1     input( 0,1 )
     1 and 0 = 1     input( 1,0 )
     1 and 1 = 0     input( 1,1 )
The task for our FeedForward Neural Network is to find the decision boundry that correctly seperates
(classifies) the given inputs as either 0 (false) or 1 (true).
Note that the provided weights are hardcoded and simulate NAND, OR and AND Gates. The network is NOT
learning them yet! */
void test3(){
    
    MultiLayerPerceptron net = MultiLayerPerceptron( {2,2,1} ) ;
    //                   NAND Gate       OR Gate      AND Gate
    net.set_weights( { {{-10,-10,15},{15,15,-10}} , {{10,10,-15}} } ) ;
    
    net.print_weights() ;
    
    std::cout << "XOR GATE:\n" ;
    std::cout <<"0 XOR 0 : "<< (((net.run( {0,0} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"0 XOR 1 : "<< (((net.run( {0,1} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"1 XOR 0 : "<< (((net.run( {1,0} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"1 XOR 1 : "<< (((net.run( {1,1} )[0]) > 0.5) ? 1 : 0) << std::endl ;
}
//....................................................................................................
void test4(){
    
    //MultiLayerPerceptron net = MultiLayerPerceptron( {2,2,1} ) ;
    //MultiLayerPerceptron net = MultiLayerPerceptron( {2,2,1},0.5,ACTIVATION::TanH ) ;
    MultiLayerPerceptron net = MultiLayerPerceptron( {2,2,1},0.5,ACTIVATION::ReLu ) ;
    
    double MSE = 0.0 ;
    int epochs = 3000 ;
    
    for( int epoch=0 ; epoch < epochs ; ++epoch ){
            
        MSE += net.back_prop( {0,0},{0} ) ;
        MSE += net.back_prop( {0,1},{1} ) ;
        MSE += net.back_prop( {1,0},{1} ) ;
        MSE += net.back_prop( {1,1},{0} ) ;
        
        MSE /= 4.0 ;
        
        if( epoch % 100 == 0 )
            std::cout <<"Epoch: "<<epoch<<" MSE: "<<MSE<< std::endl ;
    }
    std::cout << "\nBelow are the trained/learned weights:" ;
    net.print_weights() ;
    
    std::cout << "XOR GATE:\n" ;
    std::cout <<"0 XOR 0 : "<< (((net.run( {0,0} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"0 XOR 1 : "<< (((net.run( {0,1} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"1 XOR 0 : "<< (((net.run( {1,0} )[0]) > 0.5) ? 1 : 0) << std::endl ;
    std::cout <<"1 XOR 1 : "<< (((net.run( {1,1} )[0]) > 0.5) ? 1 : 0) << std::endl ;
}
//....................................................................................................
int main() {
    
    srand( u_int( time(NULL) ) ) ; // srand() uses the current time as seed for random generator rand()
    rand() ;
    
    //test0() ;
    //test0_1() ;
    //test1() ;
    //test2() ;
    /* If you noticed the pattern, while implementing AND, OR and NAND gates ALL we changed were the
    weights */
    //test3() ;
    test4() ;
    
    return 0 ;
}//...................................................................................................



