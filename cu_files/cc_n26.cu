#include <stdio.h>
#include "bmlqsim.cuh"
//Use the SVSim namespace to enable C++/CUDA APIs
using namespace SVSim;

void prepare_circuit(Simulation &sim)
{
	sim.append(Simulation::H(0));
	sim.append(Simulation::H(1));
	sim.append(Simulation::H(2));
	sim.append(Simulation::H(3));
	sim.append(Simulation::H(4));
	sim.append(Simulation::H(5));
	sim.append(Simulation::H(6));
	sim.append(Simulation::H(7));
	sim.append(Simulation::H(8));
	sim.append(Simulation::H(9));
	sim.append(Simulation::H(10));
	sim.append(Simulation::H(11));
	sim.append(Simulation::H(12));
	sim.append(Simulation::H(13));
	sim.append(Simulation::H(14));
	sim.append(Simulation::H(15));
	sim.append(Simulation::H(16));
	sim.append(Simulation::H(17));
	sim.append(Simulation::H(18));
	sim.append(Simulation::H(19));
	sim.append(Simulation::H(20));
	sim.append(Simulation::H(21));
	sim.append(Simulation::H(22));
	sim.append(Simulation::H(23));
	sim.append(Simulation::H(24));
	sim.append(Simulation::CX(0, 25));
	sim.append(Simulation::CX(1, 25));
	sim.append(Simulation::CX(2, 25));
	sim.append(Simulation::CX(3, 25));
	sim.append(Simulation::CX(4, 25));
	sim.append(Simulation::CX(5, 25));
	sim.append(Simulation::CX(6, 25));
	sim.append(Simulation::CX(7, 25));
	sim.append(Simulation::CX(8, 25));
	sim.append(Simulation::CX(9, 25));
	sim.append(Simulation::CX(10, 25));
	sim.append(Simulation::CX(11, 25));
	sim.append(Simulation::CX(12, 25));
	sim.append(Simulation::CX(13, 25));
	sim.append(Simulation::CX(14, 25));
	sim.append(Simulation::CX(15, 25));
	sim.append(Simulation::CX(16, 25));
	sim.append(Simulation::CX(17, 25));
	sim.append(Simulation::CX(18, 25));
	sim.append(Simulation::CX(19, 25));
	sim.append(Simulation::CX(20, 25));
	sim.append(Simulation::CX(21, 25));
	sim.append(Simulation::CX(22, 25));
	sim.append(Simulation::CX(23, 25));
	sim.append(Simulation::CX(24, 25));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=26;
	int n_gpus=2;
	int chunkSize=21;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
