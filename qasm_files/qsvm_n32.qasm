OPENQASM 2.0;
include "qelib1.inc";
qreg q0[32];
creg c0[32];
creg meas[32];
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
h q0[28];
h q0[29];
h q0[30];
h q0[31];
u1(1.6038498) q0[0];
u1(1.0485427) q0[1];
u1(0.40423) q0[2];
u1(2.5152579) q0[3];
u1(1.621544) q0[4];
u1(0.33195536) q0[5];
u1(1.114746) q0[6];
u1(0.24676399) q0[7];
u1(3.0442893) q0[8];
u1(0.50643241) q0[9];
u1(0.13535734) q0[10];
u1(2.2468305) q0[11];
u1(1.0802584) q0[12];
u1(1.7537201) q0[13];
u1(1.8174775) q0[14];
u1(1.9377797) q0[15];
u1(2.0411707) q0[16];
u1(2.3137966) q0[17];
u1(2.5868047) q0[18];
u1(2.0661731) q0[19];
u1(0.85578987) q0[20];
u1(0.20384853) q0[21];
u1(3.137668) q0[22];
u1(0.50249985) q0[23];
u1(0.6663842) q0[24];
u1(0.13566084) q0[25];
u1(2.8250157) q0[26];
u1(0.17638554) q0[27];
u1(0.78400627) q0[28];
u1(2.7672105) q0[29];
u1(0.29386931) q0[30];
u1(0.5744004) q0[31];
cx q0[0],q0[1];
rz(0.84565862) q0[1];
cx q0[0],q0[1];
cx q0[1],q0[2];
rz(0.20670148) q0[2];
cx q0[1],q0[2];
cx q0[2],q0[3];
rz(2.9813781) q0[3];
cx q0[2],q0[3];
cx q0[3],q0[4];
rz(1.3192529) q0[4];
cx q0[3],q0[4];
cx q0[4],q0[5];
rz(1.2898158) q0[5];
cx q0[4],q0[5];
cx q0[5],q0[6];
rz(0.32191063) q0[6];
cx q0[5],q0[6];
cx q0[6],q0[7];
rz(2.1993379) q0[7];
cx q0[6],q0[7];
cx q0[7],q0[8];
rz(0.96462255) q0[8];
cx q0[7],q0[8];
cx q0[8],q0[9];
rz(0.7183701) q0[9];
cx q0[8],q0[9];
cx q0[9],q0[10];
rz(0.0039601587) q0[10];
cx q0[9],q0[10];
cx q0[10],q0[11];
rz(0.90300351) q0[11];
cx q0[10],q0[11];
cx q0[11],q0[12];
rz(2.2209328) q0[12];
cx q0[11],q0[12];
cx q0[12],q0[13];
rz(2.927858) q0[13];
cx q0[12],q0[13];
cx q0[13],q0[14];
rz(1.017836) q0[14];
cx q0[13],q0[14];
cx q0[14],q0[15];
rz(1.4434924) q0[15];
cx q0[14],q0[15];
cx q0[15],q0[16];
rz(0.3524348) q0[16];
cx q0[15],q0[16];
cx q0[16],q0[17];
rz(0.56652442) q0[17];
cx q0[16],q0[17];
cx q0[17],q0[18];
rz(0.79715733) q0[18];
cx q0[17],q0[18];
cx q0[18],q0[19];
rz(0.22932746) q0[19];
cx q0[18],q0[19];
cx q0[19],q0[20];
rz(2.1126302) q0[20];
cx q0[19],q0[20];
cx q0[20],q0[21];
rz(3.1220866) q0[21];
cx q0[20],q0[21];
cx q0[21],q0[22];
rz(2.1102662) q0[22];
cx q0[21],q0[22];
cx q0[22],q0[23];
rz(2.5685652) q0[23];
cx q0[22],q0[23];
cx q0[23],q0[24];
rz(1.906577) q0[24];
cx q0[23],q0[24];
cx q0[24],q0[25];
rz(1.9447788) q0[25];
cx q0[24],q0[25];
cx q0[25],q0[26];
rz(3.0248347) q0[26];
cx q0[25],q0[26];
cx q0[26],q0[27];
rz(0.50556934) q0[27];
cx q0[26],q0[27];
cx q0[27],q0[28];
rz(0.18676911) q0[28];
cx q0[27],q0[28];
cx q0[28],q0[29];
rz(1.4029462) q0[29];
cx q0[28],q0[29];
cx q0[29],q0[30];
rz(0.63976518) q0[30];
cx q0[29],q0[30];
cx q0[30],q0[31];
rz(0.83269926) q0[31];
cx q0[30],q0[31];
cx q0[30],q0[31];
rz(1.556835) q0[31];
cx q0[30],q0[31];
cx q0[29],q0[30];
rz(1.1037512) q0[30];
cx q0[29],q0[30];
cx q0[28],q0[29];
rz(0.2762759) q0[29];
cx q0[28],q0[29];
cx q0[27],q0[28];
rz(0.56185333) q0[28];
cx q0[27],q0[28];
cx q0[26],q0[27];
rz(1.6192862) q0[27];
cx q0[26],q0[27];
cx q0[25],q0[26];
rz(3.0987718) q0[26];
cx q0[25],q0[26];
cx q0[24],q0[25];
rz(0.39055157) q0[25];
cx q0[24],q0[25];
cx q0[23],q0[24];
rz(2.5844281) q0[24];
cx q0[23],q0[24];
cx q0[22],q0[23];
rz(2.1802663) q0[23];
cx q0[22],q0[23];
cx q0[21],q0[22];
rz(0.58790473) q0[22];
cx q0[21],q0[22];
cx q0[20],q0[21];
rz(0.42872865) q0[21];
cx q0[20],q0[21];
cx q0[19],q0[20];
rz(0.89670381) q0[20];
cx q0[19],q0[20];
cx q0[18],q0[19];
rz(2.8069331) q0[19];
cx q0[18],q0[19];
cx q0[17],q0[18];
rz(0.30696294) q0[18];
cx q0[17],q0[18];
cx q0[16],q0[17];
rz(0.86342331) q0[17];
cx q0[16],q0[17];
cx q0[15],q0[16];
rz(0.16398037) q0[16];
cx q0[15],q0[16];
cx q0[14],q0[15];
rz(1.9205971) q0[15];
cx q0[14],q0[15];
cx q0[13],q0[14];
rz(0.14945735) q0[14];
cx q0[13],q0[14];
cx q0[12],q0[13];
rz(0.17382318) q0[13];
cx q0[12],q0[13];
cx q0[11],q0[12];
rz(0.54807591) q0[12];
cx q0[11],q0[12];
cx q0[10],q0[11];
rz(0.14470628) q0[11];
cx q0[10],q0[11];
cx q0[9],q0[10];
rz(1.6132533) q0[10];
cx q0[9],q0[10];
cx q0[8],q0[9];
rz(0.3437386) q0[9];
cx q0[8],q0[9];
cx q0[7],q0[8];
rz(0.60677659) q0[8];
cx q0[7],q0[8];
cx q0[6],q0[7];
rz(0.89676733) q0[7];
cx q0[6],q0[7];
cx q0[5],q0[6];
rz(2.5923344) q0[6];
cx q0[5],q0[6];
cx q0[4],q0[5];
rz(2.570556) q0[5];
cx q0[4],q0[5];
cx q0[3],q0[4];
rz(1.2507047) q0[4];
cx q0[3],q0[4];
cx q0[2],q0[3];
rz(1.0499484) q0[3];
cx q0[2],q0[3];
cx q0[1],q0[2];
rz(0.94790226) q0[2];
cx q0[1],q0[2];
cx q0[0],q0[1];
rz(0.075658089) q0[1];
cx q0[0],q0[1];
rz(2.2444619) q0[0];
rz(1.8417527) q0[1];
rz(2.4635264) q0[2];
rz(2.6720138) q0[3];
rz(0.14284556) q0[4];
rz(0.53004082) q0[5];
rz(2.5609071) q0[6];
rz(2.7914051) q0[7];
rz(0.79014335) q0[8];
rz(1.8631362) q0[9];
rz(1.2035721) q0[10];
rz(0.40878468) q0[11];
rz(0.96921129) q0[12];
rz(1.458419) q0[13];
rz(0.27545952) q0[14];
rz(0.86230143) q0[15];
rz(2.4062267) q0[16];
rz(1.3898969) q0[17];
rz(2.7556761) q0[18];
rz(2.5787385) q0[19];
rz(1.2722026) q0[20];
rz(2.8788487) q0[21];
rz(1.2561979) q0[22];
rz(2.4624398) q0[23];
rz(0.27176477) q0[24];
rz(1.2474856) q0[25];
rz(2.6349191) q0[26];
rz(3.0916427) q0[27];
rz(1.2716245) q0[28];
rz(2.371502) q0[29];
rz(1.1421818) q0[30];
rz(2.7555909) q0[31];
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
h q0[28];
h q0[29];
h q0[30];
h q0[31];
barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12],q0[13],q0[14],q0[15],q0[16],q0[17],q0[18],q0[19],q0[20],q0[21],q0[22],q0[23],q0[24],q0[25],q0[26],q0[27],q0[28],q0[29],q0[30],q0[31];
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
measure q0[28] -> meas[28];
measure q0[29] -> meas[29];
measure q0[30] -> meas[30];
measure q0[31] -> meas[31];
