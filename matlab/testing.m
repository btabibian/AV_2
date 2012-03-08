data2 = [0.411569071796,-0.332253492574,2
-3.84190504422,-2.6505205399,2
-0.177918621619,-0.219481279103,2
0.357821836295,-0.651736757258,2
2.16969837028,-0.454884946054,2
-1.37099157871,0.342150611973,2
1.47204529157,-0.265430824879,2
1.93757464511,-0.364843792125,2
];

data1 = [
0.0632194952932,1.80761066696,1
-0.180786876951,1.16908846187,1
1.68334966921,0.756840801972,1
3.61272269573,-1.02271002731,1
-0.495764040953,0.910726215075,1
-0.227830792622,2.81996851992,1
1.56487486219,-0.433344277091,1
2.33754647224,-0.183526067864,1
];

data0 = [
0.327506075416,-0.634107825976,0
-0.395046606739,-0.0444641137815,0
-4.24456722833,0.268279728189,0
-2.44767995717,0.200700432446,0
-0.358777672918,-0.0443407251138,0
-2.87292415213,0.100775372898,0
0.0109664382934,-0.350025474084,0
0.665297648923,-0.72447066819,0
];

figure('Name', 'Image class versus two principal components')
scatter(data0(:,1), data0(:,2), 'k+')
hold on
scatter(data1(:,1), data1(:,2), 'ro')
hold on
scatter(data2(:,1), data2(:,2), 'gx')