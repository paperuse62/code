OPENQASM 2.0;
include "qelib1.inc";
qreg q0[31];
creg c0[31];
creg meas[31];
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
u1(1.9562993) q0[0];
u1(0.73328779) q0[1];
u1(0.22379837) q0[2];
u1(3.0989792) q0[3];
u1(2.2114957) q0[4];
u1(2.8277939) q0[5];
u1(0.77077607) q0[6];
u1(0.441887) q0[7];
u1(2.4193791) q0[8];
u1(2.3707099) q0[9];
u1(0.29910407) q0[10];
u1(1.2196925) q0[11];
u1(2.7478969) q0[12];
u1(1.405387) q0[13];
u1(0.63217955) q0[14];
u1(2.4890501) q0[15];
u1(1.5928727) q0[16];
u1(0.037739964) q0[17];
u1(1.0701487) q0[18];
u1(3.0189906) q0[19];
u1(1.2993241) q0[20];
u1(1.6573938) q0[21];
u1(1.4547905) q0[22];
u1(1.6325587) q0[23];
u1(2.3296089) q0[24];
u1(0.85841936) q0[25];
u1(2.4447429) q0[26];
u1(3.1327961) q0[27];
u1(0.53338878) q0[28];
u1(2.2470261) q0[29];
u1(0.33189953) q0[30];
cx q0[0],q0[1];
rz(1.0862594) q0[1];
cx q0[0],q0[1];
cx q0[1],q0[2];
rz(0.4633627) q0[2];
cx q0[1],q0[2];
cx q0[2],q0[3];
rz(1.6847453) q0[3];
cx q0[2],q0[3];
cx q0[3],q0[4];
rz(1.2667437) q0[4];
cx q0[3],q0[4];
cx q0[4],q0[5];
rz(2.8675242) q0[5];
cx q0[4],q0[5];
cx q0[5],q0[6];
rz(2.9795827) q0[6];
cx q0[5],q0[6];
cx q0[6],q0[7];
rz(0.46336923) q0[7];
cx q0[6],q0[7];
cx q0[7],q0[8];
rz(1.9036151) q0[8];
cx q0[7],q0[8];
cx q0[8],q0[9];
rz(0.6019546) q0[9];
cx q0[8],q0[9];
cx q0[9],q0[10];
rz(0.26084578) q0[10];
cx q0[9],q0[10];
cx q0[10],q0[11];
rz(2.2900908) q0[11];
cx q0[10],q0[11];
cx q0[11],q0[12];
rz(1.1502704) q0[12];
cx q0[11],q0[12];
cx q0[12],q0[13];
rz(0.61014403) q0[13];
cx q0[12],q0[13];
cx q0[13],q0[14];
rz(1.6282012) q0[14];
cx q0[13],q0[14];
cx q0[14],q0[15];
rz(2.6946423) q0[15];
cx q0[14],q0[15];
cx q0[15],q0[16];
rz(1.2310457) q0[16];
cx q0[15],q0[16];
cx q0[16],q0[17];
rz(2.6763455) q0[17];
cx q0[16],q0[17];
cx q0[17],q0[18];
rz(1.6115006) q0[18];
cx q0[17],q0[18];
cx q0[18],q0[19];
rz(1.2151948) q0[19];
cx q0[18],q0[19];
cx q0[19],q0[20];
rz(0.77630871) q0[20];
cx q0[19],q0[20];
cx q0[20],q0[21];
rz(1.8521953) q0[21];
cx q0[20],q0[21];
cx q0[21],q0[22];
rz(0.32119255) q0[22];
cx q0[21],q0[22];
cx q0[22],q0[23];
rz(2.7104039) q0[23];
cx q0[22],q0[23];
cx q0[23],q0[24];
rz(2.4208625) q0[24];
cx q0[23],q0[24];
cx q0[24],q0[25];
rz(1.4072715) q0[25];
cx q0[24],q0[25];
cx q0[25],q0[26];
rz(1.4568729) q0[26];
cx q0[25],q0[26];
cx q0[26],q0[27];
rz(1.8323993) q0[27];
cx q0[26],q0[27];
cx q0[27],q0[28];
rz(1.8493958) q0[28];
cx q0[27],q0[28];
cx q0[28],q0[29];
rz(1.0654095) q0[29];
cx q0[28],q0[29];
cx q0[29],q0[30];
rz(1.8754801) q0[30];
cx q0[29],q0[30];
cx q0[29],q0[30];
rz(0.66007763) q0[30];
cx q0[29],q0[30];
cx q0[28],q0[29];
rz(0.70304601) q0[29];
cx q0[28],q0[29];
cx q0[27],q0[28];
rz(2.0287761) q0[28];
cx q0[27],q0[28];
cx q0[26],q0[27];
rz(1.3779363) q0[27];
cx q0[26],q0[27];
cx q0[25],q0[26];
rz(0.80465334) q0[26];
cx q0[25],q0[26];
cx q0[24],q0[25];
rz(0.31730831) q0[25];
cx q0[24],q0[25];
cx q0[23],q0[24];
rz(0.89589732) q0[24];
cx q0[23],q0[24];
cx q0[22],q0[23];
rz(2.2908596) q0[23];
cx q0[22],q0[23];
cx q0[21],q0[22];
rz(0.04005897) q0[22];
cx q0[21],q0[22];
cx q0[20],q0[21];
rz(1.3608778) q0[21];
cx q0[20],q0[21];
cx q0[19],q0[20];
rz(2.3590439) q0[20];
cx q0[19],q0[20];
cx q0[18],q0[19];
rz(2.7669783) q0[19];
cx q0[18],q0[19];
cx q0[17],q0[18];
rz(0.15945587) q0[18];
cx q0[17],q0[18];
cx q0[16],q0[17];
rz(2.3741989) q0[17];
cx q0[16],q0[17];
cx q0[15],q0[16];
rz(0.037350508) q0[16];
cx q0[15],q0[16];
cx q0[14],q0[15];
rz(0.38429344) q0[15];
cx q0[14],q0[15];
cx q0[13],q0[14];
rz(2.7923064) q0[14];
cx q0[13],q0[14];
cx q0[12],q0[13];
rz(2.9252346) q0[13];
cx q0[12],q0[13];
cx q0[11],q0[12];
rz(2.5753822) q0[12];
cx q0[11],q0[12];
cx q0[10],q0[11];
rz(0.71574249) q0[11];
cx q0[10],q0[11];
cx q0[9],q0[10];
rz(1.5622488) q0[10];
cx q0[9],q0[10];
cx q0[8],q0[9];
rz(2.0716481) q0[9];
cx q0[8],q0[9];
cx q0[7],q0[8];
rz(0.3646976) q0[8];
cx q0[7],q0[8];
cx q0[6],q0[7];
rz(3.0893929) q0[7];
cx q0[6],q0[7];
cx q0[5],q0[6];
rz(2.7074579) q0[6];
cx q0[5],q0[6];
cx q0[4],q0[5];
rz(3.0235324) q0[5];
cx q0[4],q0[5];
cx q0[3],q0[4];
rz(0.82507689) q0[4];
cx q0[3],q0[4];
cx q0[2],q0[3];
rz(2.1805395) q0[3];
cx q0[2],q0[3];
cx q0[1],q0[2];
rz(0.92234201) q0[2];
cx q0[1],q0[2];
cx q0[0],q0[1];
rz(3.0948184) q0[1];
cx q0[0],q0[1];
rz(2.0276958) q0[0];
rz(0.23452336) q0[1];
rz(2.6422606) q0[2];
rz(2.9613405) q0[3];
rz(1.2337297) q0[4];
rz(0.032709647) q0[5];
rz(1.428484) q0[6];
rz(0.50431737) q0[7];
rz(2.9215034) q0[8];
rz(2.9599728) q0[9];
rz(1.2644601) q0[10];
rz(0.16774564) q0[11];
rz(1.0704174) q0[12];
rz(2.058833) q0[13];
rz(0.66372776) q0[14];
rz(0.55983632) q0[15];
rz(2.0645378) q0[16];
rz(2.3241408) q0[17];
rz(0.59195578) q0[18];
rz(1.4118248) q0[19];
rz(2.9987079) q0[20];
rz(2.4537763) q0[21];
rz(1.8378484) q0[22];
rz(2.9374984) q0[23];
rz(0.82944626) q0[24];
rz(2.5424567) q0[25];
rz(2.4495892) q0[26];
rz(2.5961572) q0[27];
rz(3.0691211) q0[28];
rz(2.7284737) q0[29];
rz(0.55449815) q0[30];
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
barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12],q0[13],q0[14],q0[15],q0[16],q0[17],q0[18],q0[19],q0[20],q0[21],q0[22],q0[23],q0[24],q0[25],q0[26],q0[27],q0[28],q0[29],q0[30];
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