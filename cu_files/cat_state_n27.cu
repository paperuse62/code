#include <stdio.h>
#include "bmlqsim.cuh"
//Use the SVSim namespace to enable C++/CUDA APIs
using namespace SVSim;

void prepare_circuit(Simulation &sim)
{
	sim.append(Simulation::H(0));
	sim.append(Simulation::CX(0, 1));
	sim.append(Simulation::CX(1, 2));
	sim.append(Simulation::CX(2, 3));
	sim.append(Simulation::CX(3, 4));
	sim.append(Simulation::CX(4, 5));
	sim.append(Simulation::CX(5, 6));
	sim.append(Simulation::CX(6, 7));
	sim.append(Simulation::CX(7, 8));
	sim.append(Simulation::CX(8, 9));
	sim.append(Simulation::CX(9, 10));
	sim.append(Simulation::CX(10, 11));
	sim.append(Simulation::CX(11, 12));
	sim.append(Simulation::CX(12, 13));
	sim.append(Simulation::CX(13, 14));
	sim.append(Simulation::CX(14, 15));
	sim.append(Simulation::CX(15, 16));
	sim.append(Simulation::CX(16, 17));
	sim.append(Simulation::CX(17, 18));
	sim.append(Simulation::CX(18, 19));
	sim.append(Simulation::CX(19, 20));
	sim.append(Simulation::CX(20, 21));
	sim.append(Simulation::CX(21, 22));
	sim.append(Simulation::CX(22, 23));
	sim.append(Simulation::CX(23, 24));
	sim.append(Simulation::CX(24, 25));
	sim.append(Simulation::CX(25, 26));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=27;
	int n_gpus=2;
	int chunkSize=22;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
