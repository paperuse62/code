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
	sim.append(Simulation::CX(0, 24));
	sim.append(Simulation::CX(1, 24));
	sim.append(Simulation::CX(2, 24));
	sim.append(Simulation::CX(3, 24));
	sim.append(Simulation::CX(4, 24));
	sim.append(Simulation::CX(5, 24));
	sim.append(Simulation::CX(6, 24));
	sim.append(Simulation::CX(7, 24));
	sim.append(Simulation::CX(8, 24));
	sim.append(Simulation::CX(9, 24));
	sim.append(Simulation::CX(10, 24));
	sim.append(Simulation::CX(11, 24));
	sim.append(Simulation::CX(12, 24));
	sim.append(Simulation::CX(13, 24));
	sim.append(Simulation::CX(14, 24));
	sim.append(Simulation::CX(15, 24));
	sim.append(Simulation::CX(16, 24));
	sim.append(Simulation::CX(17, 24));
	sim.append(Simulation::CX(18, 24));
	sim.append(Simulation::CX(19, 24));
	sim.append(Simulation::CX(20, 24));
	sim.append(Simulation::CX(21, 24));
	sim.append(Simulation::CX(22, 24));
	sim.append(Simulation::CX(23, 24));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=25;
	int n_gpus=2;
	int chunkSize=20;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
