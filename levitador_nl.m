function [ dsdt ] = levitador_nl(t, e, u, p)

% Parametros
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

% Entradas
i1 = u(1);
i2 = u(2);

% Estados
y1 = e(1);
v1 = e(2);
y2 = e(3);
v2 = e(4);

% EDO's - Nao linear
dy1 = v1;
dv1 = (1/m1)*(i1/(a*(y1+b)^N) - i2/(a*(yc-y1+b)^N) - c/(a*(y2-y1+d)^N) - b1*v1 - m1*g);
dy2 = v2;
dv2 = (1/m2)*(i2/(a*(y2+b)^N) - i1/(a*(y2+b)^N) + c/(a*(y2-y1+d)^N) - b2*v2 - m2*g);

dsdt = [dy1; dv1; dy2; dv2];
end



