OPENQASM 2.0;
include "qelib1.inc";
qreg q[31];
creg meas[31];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cx q[0],q[2];
u1(9.0564946) q[2];
cx q[0],q[2];
u3(pi/2,0,pi) q[3];
cx q[0],q[3];
u1(9.0564946) q[3];
cx q[0],q[3];
cx q[2],q[3];
u1(9.0564946) q[3];
cx q[2],q[3];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u1(9.0564946) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[5];
cx q[3],q[5];
u1(9.0564946) q[5];
cx q[3],q[5];
u3(pi/2,0,pi) q[6];
cx q[1],q[6];
u1(9.0564946) q[6];
cx q[1],q[6];
cx q[3],q[6];
u1(9.0564946) q[6];
cx q[3],q[6];
cx q[5],q[6];
u1(9.0564946) q[6];
cx q[5],q[6];
u3(pi/2,0,pi) q[7];
cx q[3],q[7];
u1(9.0564946) q[7];
cx q[3],q[7];
cx q[4],q[7];
u1(9.0564946) q[7];
cx q[4],q[7];
cx q[6],q[7];
u1(9.0564946) q[7];
cx q[6],q[7];
u3(pi/2,0,pi) q[8];
cx q[0],q[8];
u1(9.0564946) q[8];
cx q[0],q[8];
cx q[5],q[8];
u1(9.0564946) q[8];
cx q[5],q[8];
cx q[6],q[8];
u1(9.0564946) q[8];
cx q[6],q[8];
cx q[7],q[8];
u1(9.0564946) q[8];
cx q[7],q[8];
u3(pi/2,0,pi) q[9];
cx q[0],q[9];
u1(9.0564946) q[9];
cx q[0],q[9];
cx q[7],q[9];
u1(9.0564946) q[9];
cx q[7],q[9];
cx q[8],q[9];
u1(9.0564946) q[9];
cx q[8],q[9];
u3(pi/2,0,pi) q[10];
cx q[0],q[10];
u1(9.0564946) q[10];
cx q[0],q[10];
cx q[2],q[10];
u1(9.0564946) q[10];
cx q[2],q[10];
cx q[3],q[10];
u1(9.0564946) q[10];
cx q[3],q[10];
cx q[4],q[10];
u1(9.0564946) q[10];
cx q[4],q[10];
cx q[5],q[10];
u1(9.0564946) q[10];
cx q[5],q[10];
cx q[6],q[10];
u1(9.0564946) q[10];
cx q[6],q[10];
cx q[7],q[10];
u1(9.0564946) q[10];
cx q[7],q[10];
cx q[8],q[10];
u1(9.0564946) q[10];
cx q[8],q[10];
cx q[9],q[10];
u1(9.0564946) q[10];
cx q[9],q[10];
u3(pi/2,0,pi) q[11];
cx q[0],q[11];
u1(9.0564946) q[11];
cx q[0],q[11];
cx q[2],q[11];
u1(9.0564946) q[11];
cx q[2],q[11];
cx q[3],q[11];
u1(9.0564946) q[11];
cx q[3],q[11];
cx q[5],q[11];
u1(9.0564946) q[11];
cx q[5],q[11];
cx q[6],q[11];
u1(9.0564946) q[11];
cx q[6],q[11];
cx q[7],q[11];
u1(9.0564946) q[11];
cx q[7],q[11];
cx q[9],q[11];
u1(9.0564946) q[11];
cx q[9],q[11];
u3(pi/2,0,pi) q[12];
cx q[2],q[12];
u1(9.0564946) q[12];
cx q[2],q[12];
cx q[3],q[12];
u1(9.0564946) q[12];
cx q[3],q[12];
cx q[10],q[12];
u1(9.0564946) q[12];
cx q[10],q[12];
cx q[11],q[12];
u1(9.0564946) q[12];
cx q[11],q[12];
u3(pi/2,0,pi) q[13];
cx q[0],q[13];
u1(9.0564946) q[13];
cx q[0],q[13];
cx q[1],q[13];
u1(9.0564946) q[13];
cx q[1],q[13];
cx q[2],q[13];
u1(9.0564946) q[13];
cx q[2],q[13];
cx q[7],q[13];
u1(9.0564946) q[13];
cx q[7],q[13];
cx q[9],q[13];
u1(9.0564946) q[13];
cx q[9],q[13];
cx q[10],q[13];
u1(9.0564946) q[13];
cx q[10],q[13];
cx q[11],q[13];
u1(9.0564946) q[13];
cx q[11],q[13];
u3(pi/2,0,pi) q[14];
cx q[0],q[14];
u1(9.0564946) q[14];
cx q[0],q[14];
cx q[1],q[14];
u1(9.0564946) q[14];
cx q[1],q[14];
cx q[2],q[14];
u1(9.0564946) q[14];
cx q[2],q[14];
cx q[5],q[14];
u1(9.0564946) q[14];
cx q[5],q[14];
cx q[6],q[14];
u1(9.0564946) q[14];
cx q[6],q[14];
cx q[10],q[14];
u1(9.0564946) q[14];
cx q[10],q[14];
cx q[13],q[14];
u1(9.0564946) q[14];
cx q[13],q[14];
u3(pi/2,0,pi) q[15];
cx q[1],q[15];
u1(9.0564946) q[15];
cx q[1],q[15];
cx q[4],q[15];
u1(9.0564946) q[15];
cx q[4],q[15];
cx q[5],q[15];
u1(9.0564946) q[15];
cx q[5],q[15];
cx q[8],q[15];
u1(9.0564946) q[15];
cx q[8],q[15];
cx q[11],q[15];
u1(9.0564946) q[15];
cx q[11],q[15];
cx q[12],q[15];
u1(9.0564946) q[15];
cx q[12],q[15];
cx q[13],q[15];
u1(9.0564946) q[15];
cx q[13],q[15];
u3(pi/2,0,pi) q[16];
cx q[1],q[16];
u1(9.0564946) q[16];
cx q[1],q[16];
cx q[4],q[16];
u1(9.0564946) q[16];
cx q[4],q[16];
cx q[7],q[16];
u1(9.0564946) q[16];
cx q[7],q[16];
cx q[8],q[16];
u1(9.0564946) q[16];
cx q[8],q[16];
cx q[9],q[16];
u1(9.0564946) q[16];
cx q[9],q[16];
cx q[12],q[16];
u1(9.0564946) q[16];
cx q[12],q[16];
u3(pi/2,0,pi) q[17];
cx q[0],q[17];
u1(9.0564946) q[17];
cx q[0],q[17];
cx q[1],q[17];
u1(9.0564946) q[17];
cx q[1],q[17];
cx q[2],q[17];
u1(9.0564946) q[17];
cx q[2],q[17];
cx q[3],q[17];
u1(9.0564946) q[17];
cx q[3],q[17];
cx q[4],q[17];
u1(9.0564946) q[17];
cx q[4],q[17];
cx q[6],q[17];
u1(9.0564946) q[17];
cx q[6],q[17];
cx q[12],q[17];
u1(9.0564946) q[17];
cx q[12],q[17];
cx q[14],q[17];
u1(9.0564946) q[17];
cx q[14],q[17];
u3(pi/2,0,pi) q[18];
cx q[1],q[18];
u1(9.0564946) q[18];
cx q[1],q[18];
cx q[5],q[18];
u1(9.0564946) q[18];
cx q[5],q[18];
cx q[6],q[18];
u1(9.0564946) q[18];
cx q[6],q[18];
cx q[10],q[18];
u1(9.0564946) q[18];
cx q[10],q[18];
cx q[11],q[18];
u1(9.0564946) q[18];
cx q[11],q[18];
cx q[13],q[18];
u1(9.0564946) q[18];
cx q[13],q[18];
cx q[15],q[18];
u1(9.0564946) q[18];
cx q[15],q[18];
cx q[16],q[18];
u1(9.0564946) q[18];
cx q[16],q[18];
cx q[17],q[18];
u1(9.0564946) q[18];
cx q[17],q[18];
u3(pi/2,0,pi) q[19];
cx q[1],q[19];
u1(9.0564946) q[19];
cx q[1],q[19];
cx q[2],q[19];
u1(9.0564946) q[19];
cx q[2],q[19];
cx q[6],q[19];
u1(9.0564946) q[19];
cx q[6],q[19];
cx q[10],q[19];
u1(9.0564946) q[19];
cx q[10],q[19];
cx q[14],q[19];
u1(9.0564946) q[19];
cx q[14],q[19];
cx q[17],q[19];
u1(9.0564946) q[19];
cx q[17],q[19];
cx q[18],q[19];
u1(9.0564946) q[19];
cx q[18],q[19];
u3(pi/2,0,pi) q[20];
cx q[0],q[20];
u1(9.0564946) q[20];
cx q[0],q[20];
cx q[5],q[20];
u1(9.0564946) q[20];
cx q[5],q[20];
cx q[6],q[20];
u1(9.0564946) q[20];
cx q[6],q[20];
cx q[9],q[20];
u1(9.0564946) q[20];
cx q[9],q[20];
cx q[11],q[20];
u1(9.0564946) q[20];
cx q[11],q[20];
cx q[13],q[20];
u1(9.0564946) q[20];
cx q[13],q[20];
cx q[15],q[20];
u1(9.0564946) q[20];
cx q[15],q[20];
cx q[19],q[20];
u1(9.0564946) q[20];
cx q[19],q[20];
u3(pi/2,0,pi) q[21];
cx q[1],q[21];
u1(9.0564946) q[21];
cx q[1],q[21];
cx q[2],q[21];
u1(9.0564946) q[21];
cx q[2],q[21];
cx q[3],q[21];
u1(9.0564946) q[21];
cx q[3],q[21];
cx q[4],q[21];
u1(9.0564946) q[21];
cx q[4],q[21];
cx q[5],q[21];
u1(9.0564946) q[21];
cx q[5],q[21];
cx q[6],q[21];
u1(9.0564946) q[21];
cx q[6],q[21];
cx q[8],q[21];
u1(9.0564946) q[21];
cx q[8],q[21];
cx q[11],q[21];
u1(9.0564946) q[21];
cx q[11],q[21];
cx q[16],q[21];
u1(9.0564946) q[21];
cx q[16],q[21];
cx q[17],q[21];
u1(9.0564946) q[21];
cx q[17],q[21];
cx q[19],q[21];
u1(9.0564946) q[21];
cx q[19],q[21];
u3(pi/2,0,pi) q[22];
cx q[1],q[22];
u1(9.0564946) q[22];
cx q[1],q[22];
cx q[2],q[22];
u1(9.0564946) q[22];
cx q[2],q[22];
cx q[4],q[22];
u1(9.0564946) q[22];
cx q[4],q[22];
cx q[5],q[22];
u1(9.0564946) q[22];
cx q[5],q[22];
cx q[7],q[22];
u1(9.0564946) q[22];
cx q[7],q[22];
cx q[9],q[22];
u1(9.0564946) q[22];
cx q[9],q[22];
cx q[14],q[22];
u1(9.0564946) q[22];
cx q[14],q[22];
cx q[17],q[22];
u1(9.0564946) q[22];
cx q[17],q[22];
cx q[19],q[22];
u1(9.0564946) q[22];
cx q[19],q[22];
cx q[21],q[22];
u1(9.0564946) q[22];
cx q[21],q[22];
u3(pi/2,0,pi) q[23];
cx q[0],q[23];
u1(9.0564946) q[23];
cx q[0],q[23];
cx q[1],q[23];
u1(9.0564946) q[23];
cx q[1],q[23];
cx q[2],q[23];
u1(9.0564946) q[23];
cx q[2],q[23];
cx q[3],q[23];
u1(9.0564946) q[23];
cx q[3],q[23];
cx q[7],q[23];
u1(9.0564946) q[23];
cx q[7],q[23];
cx q[9],q[23];
u1(9.0564946) q[23];
cx q[9],q[23];
cx q[12],q[23];
u1(9.0564946) q[23];
cx q[12],q[23];
cx q[16],q[23];
u1(9.0564946) q[23];
cx q[16],q[23];
cx q[19],q[23];
u1(9.0564946) q[23];
cx q[19],q[23];
u3(pi/2,0,pi) q[24];
cx q[0],q[24];
u1(9.0564946) q[24];
cx q[0],q[24];
cx q[1],q[24];
u1(9.0564946) q[24];
cx q[1],q[24];
cx q[3],q[24];
u1(9.0564946) q[24];
cx q[3],q[24];
cx q[5],q[24];
u1(9.0564946) q[24];
cx q[5],q[24];
cx q[7],q[24];
u1(9.0564946) q[24];
cx q[7],q[24];
cx q[11],q[24];
u1(9.0564946) q[24];
cx q[11],q[24];
cx q[12],q[24];
u1(9.0564946) q[24];
cx q[12],q[24];
cx q[14],q[24];
u1(9.0564946) q[24];
cx q[14],q[24];
cx q[15],q[24];
u1(9.0564946) q[24];
cx q[15],q[24];
cx q[18],q[24];
u1(9.0564946) q[24];
cx q[18],q[24];
cx q[20],q[24];
u1(9.0564946) q[24];
cx q[20],q[24];
cx q[21],q[24];
u1(9.0564946) q[24];
cx q[21],q[24];
cx q[22],q[24];
u1(9.0564946) q[24];
cx q[22],q[24];
u3(4.7115632,-pi/2,pi/2) q[24];
u3(pi/2,0,pi) q[25];
cx q[2],q[25];
u1(9.0564946) q[25];
cx q[2],q[25];
cx q[3],q[25];
u1(9.0564946) q[25];
cx q[3],q[25];
cx q[4],q[25];
u1(9.0564946) q[25];
cx q[4],q[25];
cx q[5],q[25];
u1(9.0564946) q[25];
cx q[5],q[25];
cx q[6],q[25];
u1(9.0564946) q[25];
cx q[6],q[25];
cx q[7],q[25];
u1(9.0564946) q[25];
cx q[7],q[25];
cx q[8],q[25];
u1(9.0564946) q[25];
cx q[8],q[25];
cx q[9],q[25];
u1(9.0564946) q[25];
cx q[9],q[25];
cx q[10],q[25];
u1(9.0564946) q[25];
cx q[10],q[25];
cx q[11],q[25];
u1(9.0564946) q[25];
cx q[11],q[25];
cx q[12],q[25];
u1(9.0564946) q[25];
cx q[12],q[25];
cx q[14],q[25];
u1(9.0564946) q[25];
cx q[14],q[25];
cx q[16],q[25];
u1(9.0564946) q[25];
cx q[16],q[25];
u3(4.7115632,-pi/2,pi/2) q[16];
cx q[17],q[25];
u1(9.0564946) q[25];
cx q[17],q[25];
cx q[20],q[25];
u1(9.0564946) q[25];
cx q[20],q[25];
cx q[22],q[25];
u1(9.0564946) q[25];
cx q[22],q[25];
cx q[23],q[25];
u1(9.0564946) q[25];
cx q[23],q[25];
u3(4.7115632,-pi/2,pi/2) q[23];
u3(pi/2,0,pi) q[26];
cx q[0],q[26];
u1(9.0564946) q[26];
cx q[0],q[26];
cx q[4],q[26];
u1(9.0564946) q[26];
cx q[4],q[26];
cx q[5],q[26];
u1(9.0564946) q[26];
cx q[5],q[26];
cx q[6],q[26];
u1(9.0564946) q[26];
cx q[6],q[26];
cx q[7],q[26];
u1(9.0564946) q[26];
cx q[7],q[26];
cx q[9],q[26];
u1(9.0564946) q[26];
cx q[9],q[26];
cx q[11],q[26];
u1(9.0564946) q[26];
cx q[11],q[26];
cx q[13],q[26];
u1(9.0564946) q[26];
cx q[13],q[26];
cx q[15],q[26];
u1(9.0564946) q[26];
cx q[15],q[26];
cx q[17],q[26];
u1(9.0564946) q[26];
cx q[17],q[26];
cx q[18],q[26];
u1(9.0564946) q[26];
cx q[18],q[26];
cx q[19],q[26];
u1(9.0564946) q[26];
cx q[19],q[26];
cx q[20],q[26];
u1(9.0564946) q[26];
cx q[20],q[26];
cx q[21],q[26];
u1(9.0564946) q[26];
cx q[21],q[26];
cx q[22],q[26];
u1(9.0564946) q[26];
cx q[22],q[26];
u3(4.7115632,-pi/2,pi/2) q[26];
u3(pi/2,0,pi) q[27];
cx q[0],q[27];
u1(9.0564946) q[27];
cx q[0],q[27];
cx q[4],q[27];
u1(9.0564946) q[27];
cx q[4],q[27];
cx q[6],q[27];
u1(9.0564946) q[27];
cx q[6],q[27];
cx q[7],q[27];
u1(9.0564946) q[27];
cx q[7],q[27];
cx q[8],q[27];
u1(9.0564946) q[27];
cx q[8],q[27];
cx q[10],q[27];
u1(9.0564946) q[27];
cx q[10],q[27];
cx q[11],q[27];
u1(9.0564946) q[27];
cx q[11],q[27];
cx q[14],q[27];
u1(9.0564946) q[27];
cx q[14],q[27];
cx q[17],q[27];
u1(9.0564946) q[27];
cx q[17],q[27];
cx q[18],q[27];
u1(9.0564946) q[27];
cx q[18],q[27];
cx q[19],q[27];
u1(9.0564946) q[27];
cx q[19],q[27];
cx q[20],q[27];
u1(9.0564946) q[27];
cx q[20],q[27];
cx q[21],q[27];
u1(9.0564946) q[27];
cx q[21],q[27];
u3(pi/2,0,pi) q[28];
cx q[0],q[28];
u1(9.0564946) q[28];
cx q[0],q[28];
u3(4.7115632,-pi/2,pi/2) q[0];
cx q[1],q[28];
u1(9.0564946) q[28];
cx q[1],q[28];
cx q[2],q[28];
u1(9.0564946) q[28];
cx q[2],q[28];
cx q[3],q[28];
u1(9.0564946) q[28];
cx q[3],q[28];
cx q[5],q[28];
u1(9.0564946) q[28];
cx q[5],q[28];
u3(4.7115632,-pi/2,pi/2) q[5];
cx q[8],q[28];
u1(9.0564946) q[28];
cx q[8],q[28];
u3(4.7115632,-pi/2,pi/2) q[8];
cx q[9],q[28];
u1(9.0564946) q[28];
cx q[9],q[28];
cx q[13],q[28];
u1(9.0564946) q[28];
cx q[13],q[28];
u3(4.7115632,-pi/2,pi/2) q[13];
cx q[14],q[28];
u1(9.0564946) q[28];
cx q[14],q[28];
cx q[15],q[28];
u1(9.0564946) q[28];
cx q[15],q[28];
cx q[17],q[28];
u1(9.0564946) q[28];
cx q[17],q[28];
cx q[18],q[28];
u1(9.0564946) q[28];
cx q[18],q[28];
u3(4.7115632,-pi/2,pi/2) q[18];
cx q[20],q[28];
u1(9.0564946) q[28];
cx q[20],q[28];
cx q[25],q[28];
u1(9.0564946) q[28];
cx q[25],q[28];
cx q[27],q[28];
u1(9.0564946) q[28];
cx q[27],q[28];
u3(pi/2,0,pi) q[29];
cx q[2],q[29];
u1(9.0564946) q[29];
cx q[2],q[29];
u3(4.7115632,-pi/2,pi/2) q[2];
cx q[6],q[29];
u1(9.0564946) q[29];
cx q[6],q[29];
cx q[7],q[29];
u1(9.0564946) q[29];
cx q[7],q[29];
u3(4.7115632,-pi/2,pi/2) q[7];
cx q[9],q[29];
u1(9.0564946) q[29];
cx q[9],q[29];
cx q[14],q[29];
u1(9.0564946) q[29];
cx q[14],q[29];
cx q[15],q[29];
u1(9.0564946) q[29];
cx q[15],q[29];
cx q[17],q[29];
u1(9.0564946) q[29];
cx q[17],q[29];
cx q[21],q[29];
u1(9.0564946) q[29];
cx q[21],q[29];
u3(4.7115632,-pi/2,pi/2) q[21];
cx q[22],q[29];
u1(9.0564946) q[29];
cx q[22],q[29];
cx q[25],q[29];
u1(9.0564946) q[29];
cx q[25],q[29];
cx q[27],q[29];
u1(9.0564946) q[29];
cx q[27],q[29];
cx q[28],q[29];
u1(9.0564946) q[29];
cx q[28],q[29];
u3(4.7115632,-pi/2,pi/2) q[29];
u3(4.7115632,-pi/2,pi/2) q[9];
u3(pi/2,0,pi) q[30];
cx q[1],q[30];
u1(9.0564946) q[30];
cx q[1],q[30];
u3(4.7115632,-pi/2,pi/2) q[1];
cx q[3],q[30];
u1(9.0564946) q[30];
cx q[3],q[30];
u3(4.7115632,-pi/2,pi/2) q[3];
cx q[4],q[30];
u1(9.0564946) q[30];
cx q[4],q[30];
u3(4.7115632,-pi/2,pi/2) q[4];
cx q[6],q[30];
u1(9.0564946) q[30];
cx q[6],q[30];
cx q[10],q[30];
u1(9.0564946) q[30];
cx q[10],q[30];
u3(4.7115632,-pi/2,pi/2) q[10];
cx q[11],q[30];
u1(9.0564946) q[30];
cx q[11],q[30];
u3(4.7115632,-pi/2,pi/2) q[11];
cx q[12],q[30];
u1(9.0564946) q[30];
cx q[12],q[30];
u3(4.7115632,-pi/2,pi/2) q[12];
cx q[14],q[30];
u1(9.0564946) q[30];
cx q[14],q[30];
u3(4.7115632,-pi/2,pi/2) q[14];
cx q[15],q[30];
u1(9.0564946) q[30];
cx q[15],q[30];
u3(4.7115632,-pi/2,pi/2) q[15];
cx q[17],q[30];
u1(9.0564946) q[30];
cx q[17],q[30];
u3(4.7115632,-pi/2,pi/2) q[17];
cx q[19],q[30];
u1(9.0564946) q[30];
cx q[19],q[30];
u3(4.7115632,-pi/2,pi/2) q[19];
cx q[20],q[30];
u1(9.0564946) q[30];
cx q[20],q[30];
u3(4.7115632,-pi/2,pi/2) q[20];
cx q[22],q[30];
u1(9.0564946) q[30];
cx q[22],q[30];
u3(4.7115632,-pi/2,pi/2) q[22];
cx q[25],q[30];
u1(9.0564946) q[30];
cx q[25],q[30];
u3(4.7115632,-pi/2,pi/2) q[25];
cx q[27],q[30];
u1(9.0564946) q[30];
cx q[27],q[30];
u3(4.7115632,-pi/2,pi/2) q[27];
cx q[28],q[30];
u1(9.0564946) q[30];
cx q[28],q[30];
u3(4.7115632,-pi/2,pi/2) q[28];
u3(4.7115632,-pi/2,pi/2) q[30];
u3(4.7115632,-pi/2,pi/2) q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26],q[27],q[28],q[29],q[30];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
measure q[20] -> meas[20];
measure q[21] -> meas[21];
measure q[22] -> meas[22];
measure q[23] -> meas[23];
measure q[24] -> meas[24];
measure q[25] -> meas[25];
measure q[26] -> meas[26];
measure q[27] -> meas[27];
measure q[28] -> meas[28];
measure q[29] -> meas[29];
measure q[30] -> meas[30];
