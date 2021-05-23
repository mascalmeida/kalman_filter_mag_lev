%% Projeto de Estimadores Otimos - Levitador Magnetico
% Discentes: Alice Medida, Cíntia Leal, Kelvin Ângelus
%% Filtro de Kalman Estendido
clc; clear all; close all;
%% Modelo Nao Linear
clc; clear all; close all;

% Parametros
p=[0.95 6.28 2.69 4.2 0.12 0.12 9.81 40 4 1.25 1.25];
a=p(1); 
b=p(2); 
c=p(3); 
d=p(4); 
m1=p(5); 
m2=p(6);
g=p(7);
yc=p(8);
N=p(9);
b1=p(10);
b2=p(11);

% Estado estacionario das entradas
i1 = 4500;
i2 = 9000;

% Estado estacionario das saidas
y1 = 1.6664;
v1 = 0;
y2 = 1.6995;
v2 = 0;

% Entradas
usim = [5000 11000];
u = usim' - [i1; i2];

% Condição inicial
y1_ss = 0; v1_ss = 0; y2_ss = 3; v2_ss = 0;
y0 = [y1_ss v1_ss y2_ss v2_ss];

% Definindo tempo total
ttot = 60;

% Simulacao do modelo nao linear
tsim = 0:1:(ttot-1);
[t, ymod] = ode45(@(t,e) levitador_nl(t, e, usim, p), tsim, y0);
ymod(:,1) = ymod(:,1) - y1;
ymod(:,3) = ymod(:,3) - y2;
ymod = ymod';

%% Estimador de Kalman

% Valor medido - Adicionando ruido
medicao = [1 0 0 0; 0 0 1 0];
sd = 0.2;
ruido = random('Normal', 0, sd, size(ymod));
yk = medicao*(ymod + ruido);

% Matriz de covariacia do processo
Q = 1e-5*eye(size(ymod,1));
Qn = num2str(Q(1));

% Matriz de covariancia do ruido de medicao
desv = 0.1;
R = (desv^2)*eye(size(yk,1));
Rn = num2str(R(1));

% Estimativas iniciais
x_prio = ymod(:, 1);
P0 = 1;
P0n = num2str(P0);
P_prio = P0*eye(size(ymod, 1));

%--------------------------------------------------------------------------
% Implementando o filtro de Kalman Estendido
%--------------------------------------------------------------------------
 
x_final = ymod(:, 1);
%x_2 = ymod(:, 1);
%x_3 = ymod(:, 1);

for j = 2:1:length(yk)
    [Al, Cl, r] = linearizando(p, u, x_prio);
    P_prio = Al*P_prio*Al' + Q;
    x_prio(1,:) = x_prio(1,:) + y1;
    x_prio(3,:) = x_prio(3,:) + y2;
    [t, ymod2] = ode45(@(t,e) levitador_nl(t, e, usim, p), [(j-1) j], x_prio');
    ymod2(:,1) = ymod2(:,1) - y1;
    ymod2(:,3) = ymod2(:,3) - y2;
    ymod2 = ymod2';
    x_prio = ymod2(:, end);
    erro = x_prio - ymod(:, j);
    Ke = P_prio*Cl'*inv((Cl*P_prio*Cl' + R));
    P_post = P_prio - Ke*Cl*P_prio;
    x_post = x_prio + Ke*(yk(:, j) - Cl*x_prio);
    x_final(:, j) = x_post;
    %x_2(:, j) = x_post;
    %x_3(:, j) = x_post;
    P_prio = P_post;
    x_prio = x_post;
end

%% Resultados

figure;
subplot(2,2,1)
plot(tsim, yk(1,:) + y1, 'kx', tsim, ymod(1,:) + y1, 'r:', tsim, x_final(1,:) + y1, 'b','LineWidth', 1.5)
title({"EFK - Posições","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
legend('y1 Medido', 'y1 Não Linear', 'y1 EFK');
xlabel("Tempo (s)");
ylabel("Posição (cm)");
subplot(2,2,2)
plot(tsim, ymod(2,:), 'r:', tsim, x_final(2,:), 'b', 'LineWidth', 1.5)
title({"EFK - Velocidades","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
legend('v1 Não Linear', 'v1 EFK');
xlabel("Tempo (s)");
ylabel("Velocidade (cm/s)");
subplot(2,2,3)
plot(tsim, yk(2,:) + y2, 'kx', tsim, ymod(3,:) + y2, 'r:', tsim, x_final(3,:) + y2, 'b', 'LineWidth', 1.5)
legend('y2 Medido', 'y2 Não Linear','y2 EFK');
xlabel("Tempo (s)");
ylabel("Posição (cm)");
subplot(2,2,4)
plot(tsim, ymod(4,:), 'r:', tsim, x_final(4,:), 'b', 'LineWidth', 1.5)
legend('v2 Não Linear', 'v2 EFK');
xlabel("Tempo (s)");
ylabel("Velocidade (cm/s)");

% % Graficos para analise de sensibilidade
% figure;
% subplot(2,2,1)
% plot(tsim, yk(1,:) + y1, 'kx', tsim, ymod(1,:) + y1, 'r:', tsim, x_final(1,:) + y1, 'g',tsim, x_2(1,:) + y1, 'b',tsim, x_3(1,:) + y1, 'm', 'LineWidth', 1.5)
% title({"Filtro de Kalman Estendido - Posições","Sintonia: Q = " + Qn + " e R = " + Rn});
% legend('y1 Medido', 'y1 Não Linear', 'y1 P(0) = 1', 'y1 P(0) = 0.005', 'y1 P(0) = 0.00005');
% xlabel("Tempo (s)");
% ylabel("Posição (cm)");
% subplot(2,2,2)
% plot(tsim, ymod(2,:), 'r:', tsim, x_final(2,:), 'g',tsim, x_2(2,:), 'b',tsim, x_3(2,:), 'm', 'LineWidth', 1.5)
% title({"Filtro de Kalman Estendido - Velocidades","Sintonia: Q = " + Qn + " e R = " + Rn});
% legend('v1 Não Linear', 'v1 P(0) = 1', 'v1 P(0) = 0.005', 'v1 P(0) = 0.00005');
% xlabel("Tempo (s)");
% ylabel("Velocidade (cm/s)");
% subplot(2,2,3)
% plot(tsim, yk(2,:) + y2, 'kx', tsim, ymod(3,:) + y2, 'r:', tsim, x_final(3,:) + y2, 'g',tsim, x_2(3,:) + y2, 'b',tsim, x_3(3,:) + y2, 'm', 'LineWidth', 1.5)
% legend('y2 Medido', 'y2 Não Linear', 'y2 P(0) = 1', 'y2 P(0) = 0.005', 'y2 P(0) = 0.00005');
% xlabel("Tempo (s)");
% ylabel("Posição (cm)");
% subplot(2,2,4)
% plot(tsim, ymod(4,:), 'r:', tsim, x_final(4,:), 'g',tsim, x_2(4,:), 'b',tsim, x_3(4,:), 'm', 'LineWidth', 1.5)
% legend('v2 Não Linear', 'v2 P(0) = 1', 'v2 P(0) = 0.005', 'v2 P(0) = 0.00005');
% xlabel("Tempo (s)");
% ylabel("Velocidade (cm/s)");

% % Graficos para comparação dos filtros
% figure;
% subplot(2,2,1)
% plot(tsim, abs(erro_int(1,:)), 'k', 'LineWidth', 1.5)
% title({"Filtro de Kalman Estendido - Erro Filtro das Posições","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
% legend('Posição y1');
% xlabel("Tempo (s)");
% ylabel("Erro (Posições)");
% subplot(2,2,2)
% plot(tsim, abs(erro_int(3,:)), 'm', 'LineWidth', 1.5)
% title({"Filtro de Kalman Estendido - Erro Estimação das Velocidades","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
% legend('Velocidade v1');
% xlabel("Tempo (s)");
% ylabel("Erro (Velocidades)");
% subplot(2,2,3)
% plot(tsim, abs(erro_int(2,:)), 'k', 'LineWidth', 1.5)
% legend('Posição y2');
% xlabel("Tempo (s)");
% ylabel("Erro (Posições)");
% subplot(2,2,4)
% plot(tsim, abs(erro_int(1,:)), 'm', 'LineWidth', 1.5)
% legend('Velocidade v2');
% xlabel("Tempo (s)");
% ylabel("Erro (Velocidades)");
% 
