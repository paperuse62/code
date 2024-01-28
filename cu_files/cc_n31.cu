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
	sim.append(Simulation::H(25));
	sim.append(Simulation::H(26));
	sim.append(Simulation::H(27));
	sim.append(Simulation::H(28));
	sim.append(Simulation::H(29));
	sim.append(Simulation::CX(0, 30));
	sim.append(Simulation::CX(1, 30));
	sim.append(Simulation::CX(2, 30));
	sim.append(Simulation::CX(3, 30));
	sim.append(Simulation::CX(4, 30));
	sim.append(Simulation::CX(5, 30));
	sim.append(Simulation::CX(6, 30));
	sim.append(Simulation::CX(7, 30));
	sim.append(Simulation::CX(8, 30));
	sim.append(Simulation::CX(9, 30));
	sim.append(Simulation::CX(10, 30));
	sim.append(Simulation::CX(11, 30));
	sim.append(Simulation::CX(12, 30));
	sim.append(Simulation::CX(13, 30));
	sim.append(Simulation::CX(14, 30));
	sim.append(Simulation::CX(15, 30));
	sim.append(Simulation::CX(16, 30));
	sim.append(Simulation::CX(17, 30));
	sim.append(Simulation::CX(18, 30));
	sim.append(Simulation::CX(19, 30));
	sim.append(Simulation::CX(20, 30));
	sim.append(Simulation::CX(21, 30));
	sim.append(Simulation::CX(22, 30));
	sim.append(Simulation::CX(23, 30));
	sim.append(Simulation::CX(24, 30));
	sim.append(Simulation::CX(25, 30));
	sim.append(Simulation::CX(26, 30));
	sim.append(Simulation::CX(27, 30));
	sim.append(Simulation::CX(28, 30));
	sim.append(Simulation::CX(29, 30));
}

int main()
{
	srand(RAND_SEED);
	int n_qubits=31;
	int n_gpus=2;
	int chunkSize=25;
	Simulation sim(n_qubits, n_gpus, chunkSize);
	prepare_circuit(sim);
	sim.beginSimulation();
	return 0;
}
