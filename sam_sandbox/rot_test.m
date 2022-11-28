%% testbench to test rotating 2d vector
clear all; close all; home;

%% format
set(0, 'DefaultTextFontName', 'Arial', 'DefaultTextFontSize', 20, 'DefaultTextFontWEight', 'Bold');
set(0, 'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 20, 'DefaultAxesFontWeight', 'Bold', 'DefaultAxesLineWidth', 1.5);
set(0, 'DefaultLineLineWidth', 3, 'DefaultLineMarkerSize', 10);

%% create 2d vector

v = [1; 0];
v0 = [0; 0];

%rotate the vector with yaw angle
yaw = -pi/4;

C = [cos(yaw), -sin(yaw);
     sin(yaw), cos(yaw)];

vRot = C * v;


figure, 
quiver(v0(1), v0(2), v(1), v(2), 0);
hold on;
quiver(v0(1), v0(2), vRot(1), vRot(2));


%% calculate distance
x_drone = 37; y_drone = 68;
x_targ = 360; y_targ = 328;

d_s = sqrt((x_drone-x_targ)^2 + (y_drone-y_targ)^2)


d_s_matlab = norm([(x_drone-x_targ); (y_drone-y_targ)])


%% T

dmin = 100;
dstep = 0.1;
dl = [dstep:dstep:dmin-dstep];

T = -exp(1 - dmin^2./(dmin^2 - dl.^2 + 1e-4));

figure(222); hold on;
plot( dl, T );
xlabel('$d_l$', 'interpreter','latex');
ylabel('$T$', 'interpreter','latex');
legend("$d_{min} = 100 \ [pixels]$", 'interpreter', 'latex');

