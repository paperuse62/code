OPENQASM 2.0;
include "qelib1.inc";
qreg q0[25];
creg c0[25];
h q0[0];
h q0[1];
h q0[2];
h q0[3];
h q0[4];
h q0[5];
h q0[6];
h q0[7];
h q0[8];
h q0[9];
h q0[10];
h q0[11];
h q0[12];
h q0[13];
h q0[14];
h q0[15];
h q0[16];
h q0[17];
h q0[18];
h q0[19];
h q0[20];
h q0[21];
h q0[22];
h q0[23];
cx q0[0],q0[24];
cx q0[1],q0[24];
cx q0[2],q0[24];
cx q0[3],q0[24];
cx q0[4],q0[24];
cx q0[5],q0[24];
cx q0[6],q0[24];
cx q0[7],q0[24];
cx q0[8],q0[24];
cx q0[9],q0[24];
cx q0[10],q0[24];
cx q0[11],q0[24];
cx q0[12],q0[24];
cx q0[13],q0[24];
cx q0[14],q0[24];
cx q0[15],q0[24];
cx q0[16],q0[24];
cx q0[17],q0[24];
cx q0[18],q0[24];
cx q0[19],q0[24];
cx q0[20],q0[24];
cx q0[21],q0[24];
cx q0[22],q0[24];
cx q0[23],q0[24];
measure q0[24] -> c0[24];
if(c0==0) x q0[24];
if(c0==0) h q0[24];
if(c0==16777216) h q0[0];
if(c0==16777216) h q0[1];
if(c0==16777216) h q0[2];
if(c0==16777216) h q0[3];
if(c0==16777216) h q0[4];
if(c0==16777216) h q0[5];
if(c0==16777216) h q0[6];
if(c0==16777216) h q0[7];
if(c0==16777216) h q0[8];
if(c0==16777216) h q0[9];
if(c0==16777216) h q0[10];
if(c0==16777216) h q0[11];
if(c0==16777216) h q0[12];
if(c0==16777216) h q0[13];
if(c0==16777216) h q0[14];
if(c0==16777216) h q0[15];
if(c0==16777216) h q0[16];
if(c0==16777216) h q0[17];
if(c0==16777216) h q0[18];
if(c0==16777216) h q0[19];
if(c0==16777216) h q0[20];
if(c0==16777216) h q0[21];
if(c0==16777216) h q0[22];
if(c0==16777216) h q0[23];
barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12],q0[13],q0[14],q0[15],q0[16],q0[17],q0[18],q0[19],q0[20],q0[21],q0[22],q0[23],q0[24];
if(c0==0) cx q0[6],q0[24];
barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12],q0[13],q0[14],q0[15],q0[16],q0[17],q0[18],q0[19],q0[20],q0[21],q0[22],q0[23],q0[24];
if(c0==0) h q0[0];
if(c0==0) h q0[1];
if(c0==0) h q0[2];
if(c0==0) h q0[3];
if(c0==0) h q0[4];
if(c0==0) h q0[5];
if(c0==0) h q0[6];
if(c0==0) h q0[7];
if(c0==0) h q0[8];
if(c0==0) h q0[9];
if(c0==0) h q0[10];
if(c0==0) h q0[11];
if(c0==0) h q0[12];
if(c0==0) h q0[13];
if(c0==0) h q0[14];
if(c0==0) h q0[15];
if(c0==0) h q0[16];
if(c0==0) h q0[17];
if(c0==0) h q0[18];
if(c0==0) h q0[19];
if(c0==0) h q0[20];
if(c0==0) h q0[21];
if(c0==0) h q0[22];
if(c0==0) h q0[23];
measure q0[0] -> c0[0];
measure q0[1] -> c0[1];
measure q0[2] -> c0[2];
measure q0[3] -> c0[3];
measure q0[4] -> c0[4];
measure q0[5] -> c0[5];
measure q0[6] -> c0[6];
measure q0[7] -> c0[7];
measure q0[8] -> c0[8];
measure q0[9] -> c0[9];
measure q0[10] -> c0[10];
measure q0[11] -> c0[11];
measure q0[12] -> c0[12];
measure q0[13] -> c0[13];
measure q0[14] -> c0[14];
measure q0[15] -> c0[15];
measure q0[16] -> c0[16];
measure q0[17] -> c0[17];
measure q0[18] -> c0[18];
measure q0[19] -> c0[19];
measure q0[20] -> c0[20];
measure q0[21] -> c0[21];
measure q0[22] -> c0[22];
measure q0[23] -> c0[23];
