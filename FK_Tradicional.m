%% Projeto de Estimadores Otimos - Levitador Magnetico
% Discentes: Alice Medida, Cíntia Leal, Kelvin Ângelus
%% Filtro de Kalman Tradicional
clear all; close all; clc; 
%% Linearização
clear all; close all; clc; 

% Definindo variaveis/parametros
syms y1 v1 y2 v2 i1 i2 a b c d yc N b1 b2 m1 m2 g

% EDO's - Nao linear
dy1 = v1;
dv1 = (1/m1)*(i1/(a*(y1+b)^N) - i2/(a*(yc-y1+b)^N) - c/(a*(y2-y1+d)^N) - b1*v1 - m1*g);
dy2 = v2;
dv2 = (1/m2)*(i2/(a*(y2+b)^N) - i1/(a*(y2+b)^N) + c/(a*(y2-y1+d)^N) - b2*v2 - m2*g);

% Saidas
s1 = y1;
s2 = v1;
s3 = y2;
s4 = v2;

% Vetor de EDO's
V_EDO = [dy1 dv1 dy2 dv2];

% Vetor de entradas
V_U = [i1 i2];

% Vetor de estados
V_X = [y1 v1 y2 v2];

% Vetor das saidas
V_S = [s1 s3];

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
i1=4500;
i2=9000;

% Estado estacionario das saidas
y1 = 1.6664;
v1 = 0;
y2 = 1.6995;
v2 = 0;

% Linearizacao
% Matriz jacobiana das EDO's em relacao aos estados NLinhas: Estados x NColunas: Estados
A = double(subs(jacobian(V_EDO, V_X)));
% Matriz jacobiana das EDO's em relacao as entradas NLinhas: Estados x NColunas: Entradas
B = double(subs(jacobian(V_EDO, V_U)));
% Matriz jacobiana das saidas em relacao aos estados NLinhas: Saidas vs NColunas: Estados
C = double(subs(jacobian(V_S, V_X)));
% Matriz jacobiana das saidas em relacao as entradas NLinhas: Saidas vs NColunas: Entradas
D = double(subs(jacobian(V_S, V_U)));
% Estabilidade
raizes = eig(A);
% Modelo continuo linear
sys_c = ss(A, B, C, D);

%% Discretização

% Modelo discreto linear
Ts = 0.1;
sys_d = c2d(sys_c, Ts);
Ad = sys_d.A;
Bd = sys_d.B;
Cd = sys_d.C; 

%% Estimador de Kalman

% Definindo tempo total
ttot = 60;

% Medicao da planta

% Condicoes iniciais e Desvio
y1_0 = 0; v1_0 = 0; y2_0 = 3; v2_0 = 0;
y0 = [y1_0 v1_0 y2_0 v2_0]';
usim = [5000 11000]';
u = usim;
u(1) = usim(1) - i1;
u(2) = usim(2) - i2;

% Simulacao do valor medido (nao linear)
tsim = 0:1:(ttot-1);
[t, ymod] = ode45(@(t,e) levitador_nl(t, e, usim, p), tsim, y0);
ymod(:,1) = ymod(:,1) - y1;
ymod(:,3) = ymod(:,3) - y2;
ymod = ymod';


% Adicionando ruido
sd = 0.2;
ruido = random('Normal', 0, sd, size(ymod));
yk = Cd*(ymod + ruido);

% Matriz de covariacia do processo
Q = 1e-3*eye(length(Ad));
Qn = num2str(Q(1));

% Matriz de covariancia do ruido de medicao
desv = 0.75;
R = (desv^2)*eye(size(Cd,1));
Rn = num2str(R(1));

% Estimativas iniciais
x_prio = [(y1_0-y1) v1_0 (y2_0-y2) v2_0]';
P0 = 1;
P0n = num2str(P0);
P_prio = P0*eye(length(Ad));

%--------------------------------------------------------------------------
% Implementando o filtro de Kalman
%--------------------------------------------------------------------------

x_finalFK = x_prio;

for j = 2:1:length(yk)
    P_prio = Ad*P_prio*Ad' + Q;
    x_prio = Ad*x_prio + Bd*u;
    Ke = P_prio*Cd'*inv((Cd*P_prio*Cd' + R));
    P_post = P_prio - Ke*Cd*P_prio;
    x_post = x_prio + Ke*(yk(:, j) - Cd*x_prio);
    x_finalFK(:, j) = x_post;
    erro_int(:, j) = ymod(:, j) - x_finalFK(1:4, j);
    P_prio = P_post;
    x_prio = x_post;
end

erro = ymod(:, end) - x_finalFK(1:4, end);

%% Resultados

% Tempo de simulacao
tsim = 0:1:(ttot-1);

figure;
subplot(2,2,1)
plot(tsim, yk(1,:) + y1, 'kx', tsim, ymod(1,:) + y1, 'r:', tsim, x_finalFK(1,:) + y1, 'g','LineWidth', 1.5)
title({"FK - Posições","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
legend('y1 Medido', 'y1 Não Linear', 'y1 FK');
xlabel("Tempo (s)");
ylabel("Posição (cm)");
subplot(2,2,2)
plot(tsim, ymod(2,:), 'r:', tsim, x_finalFK(2,:), 'g', 'LineWidth', 1.5)
title({"FK - Velocidades","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
legend('v1 Não Linear', 'v1 FK');
xlabel("Tempo (s)");
ylabel("Velocidade (cm/s)");
subplot(2,2,3)
plot(tsim, yk(2,:) + y2, 'kx', tsim, ymod(3,:) + y2, 'r:', tsim, x_finalFK(3,:) + y2, 'g', 'LineWidth', 1.5)
legend('y2 Medido', 'y2 Não Linear','y2 FK');
xlabel("Tempo (s)");
ylabel("Posição (cm)");
subplot(2,2,4)
plot(tsim, ymod(4,:), 'r:', tsim, x_finalFK(4,:), 'g', 'LineWidth', 1.5)
legend('v2 Não Linear', 'v2 FK');
xlabel("Tempo (s)");
ylabel("Velocidade (cm/s)");


% % Graficos para comparação FF vs EFK
% figure;
% subplot(2,2,1)
% plot(tsim, yk(1,:) + y1, 'kx', tsim, ymod(1,:) + y1, 'r:', tsim, x_finalFK(1,:) + y1, 'g', tsim, x_final(1,:) + y1, 'b','LineWidth', 1.5)
% title({"FK vs EFK - Posições","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
% legend('y1 Medido', 'y1 Não Linear', 'y1 FK', 'y1 EFK');
% xlabel("Tempo (s)");
% ylabel("Posição (cm)");
% subplot(2,2,2)
% plot(tsim, ymod(2,:), 'r:', tsim, x_finalFK(2,:), 'g', tsim, x_final(2,:), 'b', 'LineWidth', 1.5)
% title({"FK vs EFK - Velocidades","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
% legend('v1 Não Linear', 'v1 FK', 'v1 EFK');
% xlabel("Tempo (s)");
% ylabel("Velocidade (cm/s)");
% subplot(2,2,3)
% plot(tsim, yk(2,:) + y2, 'kx', tsim, ymod(3,:) + y2, 'r:', tsim, x_finalFK(3,:) + y2, 'g', tsim, x_final(3,:) + y2, 'b', 'LineWidth', 1.5)
% legend('y2 Medido', 'y2 Não Linear','y2 FK', 'y2 EFK');
% xlabel("Tempo (s)");
% ylabel("Posição (cm)");
% subplot(2,2,4)
% plot(tsim, ymod(4,:), 'r:', tsim, x_finalFK(4,:), 'g', tsim, x_final(4,:), 'b', 'LineWidth', 1.5)
% legend('v2 Não Linear', 'v2 FK', 'v2 EFK');
% xlabel("Tempo (s)");
% ylabel("Velocidade (cm/s)");

% Avaliacao do erro

% % Graficos
% figure;
% subplot(2,2,1)
% plot(tsim, abs(erro_int(1,:)), 'k', 'LineWidth', 1.5)
% title({"Filtro de Kalman Tradicional - Erro Filtro das Posições","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
% legend('Posição y1');
% xlabel("Tempo (s)");
% ylabel("Erro (Posições)");
% subplot(2,2,2)
% plot(tsim, abs(erro_int(3,:)), 'm', 'LineWidth', 1.5)
% title({"Filtro de Kalman Tradicional - Erro Estimação das Velocidades","Sintonia: Q = " + Qn + ", R = " + Rn + " e P(0) = " + P0n});
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
