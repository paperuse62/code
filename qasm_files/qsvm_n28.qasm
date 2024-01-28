OPENQASM 2.0;
include "qelib1.inc";
qreg q0[28];
creg c0[28];
creg meas[28];
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
h q0[24];
h q0[25];
h q0[26];
h q0[27];
u1(2.1699038) q0[0];
u1(0.9551779) q0[1];
u1(0.65712534) q0[2];
u1(1.8243812) q0[3];
u1(1.0437124) q0[4];
u1(1.6750941) q0[5];
u1(1.4220504) q0[6];
u1(2.3042528) q0[7];
u1(0.94833865) q0[8];
u1(2.734) q0[9];
u1(1.6241093) q0[10];
u1(2.6792377) q0[11];
u1(0.62024573) q0[12];
u1(3.0910522) q0[13];
u1(1.9235026) q0[14];
u1(1.8565519) q0[15];
u1(0.59060641) q0[16];
u1(1.338008) q0[17];
u1(1.7936934) q0[18];
u1(2.8372264) q0[19];
u1(2.0054162) q0[20];
u1(2.891353) q0[21];
u1(0.65809484) q0[22];
u1(3.0106209) q0[23];
u1(1.9730519) q0[24];
u1(0.25515122) q0[25];
u1(2.3421439) q0[26];
u1(1.2882864) q0[27];
cx q0[0],q0[1];
rz(1.3982604) q0[1];
cx q0[0],q0[1];
cx q0[1],q0[2];
rz(1.9704339) q0[2];
cx q0[1],q0[2];
cx q0[2],q0[3];
rz(0.53060757) q0[3];
cx q0[2],q0[3];
cx q0[3],q0[4];
rz(0.8083567) q0[4];
cx q0[3],q0[4];
cx q0[4],q0[5];
rz(2.2673691) q0[5];
cx q0[4],q0[5];
cx q0[5],q0[6];
rz(1.6434682) q0[6];
cx q0[5],q0[6];
cx q0[6],q0[7];
rz(0.37794015) q0[7];
cx q0[6],q0[7];
cx q0[7],q0[8];
rz(2.5840236) q0[8];
cx q0[7],q0[8];
cx q0[8],q0[9];
rz(2.7199212) q0[9];
cx q0[8],q0[9];
cx q0[9],q0[10];
rz(0.002364384) q0[10];
cx q0[9],q0[10];
cx q0[10],q0[11];
rz(0.9060271) q0[11];
cx q0[10],q0[11];
cx q0[11],q0[12];
rz(2.3435787) q0[12];
cx q0[11],q0[12];
cx q0[12],q0[13];
rz(1.3019047) q0[13];
cx q0[12],q0[13];
cx q0[13],q0[14];
rz(1.8422213) q0[14];
cx q0[13],q0[14];
cx q0[14],q0[15];
rz(1.6456156) q0[15];
cx q0[14],q0[15];
cx q0[15],q0[16];
rz(1.2348151) q0[16];
cx q0[15],q0[16];
cx q0[16],q0[17];
rz(2.1874958) q0[17];
cx q0[16],q0[17];
cx q0[17],q0[18];
rz(1.6056739) q0[18];
cx q0[17],q0[18];
cx q0[18],q0[19];
rz(2.1379703) q0[19];
cx q0[18],q0[19];
cx q0[19],q0[20];
rz(0.32229654) q0[20];
cx q0[19],q0[20];
cx q0[20],q0[21];
rz(0.58057063) q0[21];
cx q0[20],q0[21];
cx q0[21],q0[22];
rz(2.8948937) q0[22];
cx q0[21],q0[22];
cx q0[22],q0[23];
rz(0.13252027) q0[23];
cx q0[22],q0[23];
cx q0[23],q0[24];
rz(0.99534298) q0[24];
cx q0[23],q0[24];
cx q0[24],q0[25];
rz(1.6867354) q0[25];
cx q0[24],q0[25];
cx q0[25],q0[26];
rz(0.54745851) q0[26];
cx q0[25],q0[26];
cx q0[26],q0[27];
rz(0.93980975) q0[27];
cx q0[26],q0[27];
cx q0[26],q0[27];
rz(2.7643594) q0[27];
cx q0[26],q0[27];
cx q0[25],q0[26];
rz(1.3828968) q0[26];
cx q0[25],q0[26];
cx q0[24],q0[25];
rz(0.73216846) q0[25];
cx q0[24],q0[25];
cx q0[23],q0[24];
rz(0.290309) q0[24];
cx q0[23],q0[24];
cx q0[22],q0[23];
rz(2.0720902) q0[23];
cx q0[22],q0[23];
cx q0[21],q0[22];
rz(2.1078978) q0[22];
cx q0[21],q0[22];
cx q0[20],q0[21];
rz(2.5238571) q0[21];
cx q0[20],q0[21];
cx q0[19],q0[20];
rz(2.9560382) q0[20];
cx q0[19],q0[20];
cx q0[18],q0[19];
rz(0.64090617) q0[19];
cx q0[18],q0[19];
cx q0[17],q0[18];
rz(2.3683419) q0[18];
cx q0[17],q0[18];
cx q0[16],q0[17];
rz(2.2999625) q0[17];
cx q0[16],q0[17];
cx q0[15],q0[16];
rz(3.0877032) q0[16];
cx q0[15],q0[16];
cx q0[14],q0[15];
rz(2.1826298) q0[15];
cx q0[14],q0[15];
cx q0[13],q0[14];
rz(2.3459468) q0[14];
cx q0[13],q0[14];
cx q0[12],q0[13];
rz(0.51291426) q0[13];
cx q0[12],q0[13];
cx q0[11],q0[12];
rz(2.0573142) q0[12];
cx q0[11],q0[12];
cx q0[10],q0[11];
rz(2.0171579) q0[11];
cx q0[10],q0[11];
cx q0[9],q0[10];
rz(1.7333721) q0[10];
cx q0[9],q0[10];
cx q0[8],q0[9];
rz(0.11878236) q0[9];
cx q0[8],q0[9];
cx q0[7],q0[8];
rz(0.66431821) q0[8];
cx q0[7],q0[8];
cx q0[6],q0[7];
rz(1.4776484) q0[7];
cx q0[6],q0[7];
cx q0[5],q0[6];
rz(0.15992494) q0[6];
cx q0[5],q0[6];
cx q0[4],q0[5];
rz(2.6379147) q0[5];
cx q0[4],q0[5];
cx q0[3],q0[4];
rz(2.6798914) q0[4];
cx q0[3],q0[4];
cx q0[2],q0[3];
rz(0.48469111) q0[3];
cx q0[2],q0[3];
cx q0[1],q0[2];
rz(0.66204571) q0[2];
cx q0[1],q0[2];
cx q0[0],q0[1];
rz(0.38050381) q0[1];
cx q0[0],q0[1];
rz(0.017558745) q0[0];
rz(1.7146039) q0[1];
rz(1.8988783) q0[2];
rz(0.69364104) q0[3];
rz(1.3667962) q0[4];
rz(0.76886335) q0[5];
rz(2.2322675) q0[6];
rz(2.2696695) q0[7];
rz(3.0430036) q0[8];
rz(1.9467983) q0[9];
rz(1.8520253) q0[10];
rz(2.8761954) q0[11];
rz(2.1186918) q0[12];
rz(2.6340059) q0[13];
rz(2.9586964) q0[14];
rz(2.4618192) q0[15];
rz(2.6453141) q0[16];
rz(2.752641) q0[17];
rz(0.12178244) q0[18];
rz(2.2971991) q0[19];
rz(1.2692903) q0[20];
rz(0.11290433) q0[21];
rz(0.17328893) q0[22];
rz(1.3604765) q0[23];
rz(1.3851521) q0[24];
rz(0.6197422) q0[25];
rz(1.8985086) q0[26];
rz(2.7322205) q0[27];
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
h q0[24];
h q0[25];
h q0[26];
h q0[27];
barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12],q0[13],q0[14],q0[15],q0[16],q0[17],q0[18],q0[19],q0[20],q0[21],q0[22],q0[23],q0[24],q0[25],q0[26],q0[27];
measure q0[0] -> meas[0];
measure q0[1] -> meas[1];
measure q0[2] -> meas[2];
measure q0[3] -> meas[3];
measure q0[4] -> meas[4];
measure q0[5] -> meas[5];
measure q0[6] -> meas[6];
measure q0[7] -> meas[7];
measure q0[8] -> meas[8];
measure q0[9] -> meas[9];
measure q0[10] -> meas[10];
measure q0[11] -> meas[11];
measure q0[12] -> meas[12];
measure q0[13] -> meas[13];
measure q0[14] -> meas[14];
measure q0[15] -> meas[15];
measure q0[16] -> meas[16];
measure q0[17] -> meas[17];
measure q0[18] -> meas[18];
measure q0[19] -> meas[19];
measure q0[20] -> meas[20];
measure q0[21] -> meas[21];
measure q0[22] -> meas[22];
measure q0[23] -> meas[23];
measure q0[24] -> meas[24];
measure q0[25] -> meas[25];
measure q0[26] -> meas[26];
measure q0[27] -> meas[27];