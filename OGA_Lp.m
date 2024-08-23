%% approximate the function f on [0,1] with iter CGA iterations
iter = 30; p = 3;
%f = @(z) sin(2*z)+cos(z); 
%f = @(z) sin(z);
f = @(z) sin(pi*z); 
N = 500; h = 1/N;
node = (0:h:1)';

c = [1/2-sqrt(15)/10 1/2 1/2+sqrt(15)/10];% 3-point Gauss quadrature
qpt = [node(1:end-1)+c(1)*h;node(1:end-1)+c(2)*h;node(1:end-1)+c(3)*h];

hd = 5e-5; b = (-2.0:hd:2.0)'; % discretization of the dictionary
nd = length(b); 

g = [repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b']>0; % ReLU0
%g = max([repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b'],0); % ReLU1
  
fqpt = f(qpt); r = fqpt;
A = zeros(iter,iter); rhs = zeros(iter,1); % matrix and rhs for projection
id = zeros(iter,1); argmax = zeros(nd,1);
err = id; C = 0;
for i = 1:iter
    for j = 1:nd
        %argmax(j) = mynorm(g(:,j).*sign(r).*abs(r).^(p-1)); % norming 1
        argmax(j) = epnorm( g(:,j).*abs(r).^(p-2).*r ); % norming 2
    end
    [~,id(i)] = max(abs(argmax));
    %[~,id(i)] = max(argmax);
    
    obj = @(x) epnorm(abs(fqpt - g(:,id(1:i))*x).^p);
    C = fminunc(obj,zeros(i,1));
    %C = fminunc(obj,C);
    %C = fsolve(@(x) eqi(x,g(:,id(1:i)),fqpt,h,p),C);
    r = fqpt - g(:,id(1:i))*C;
    C = [C;0];
    err(i) = (h*epnorm(abs(r).^p)).^(1/p);

    fprintf("Step %d, error is %f\n",i,err(i));
end

% gamma = zeros(iter,1);
% for i=1:iter
%     X0 = zeros(iter,1);
%     X0(i) = X0(i) + 1/sqrt( h*mynorm( g(:,id(i)).^2 ) );
%     X = fmincon(@(x) h*mynorm(abs(g(:,id)*x).^p),X0,[],[],[],[],[],[],@(x) mycon(x,g(:,id),h));
%     gamma(i) = ( h*mynorm( abs(g(:,id)*X).^p) ).^(1/p)/( h*mynorm(abs(g(:,id)*X).^2) ).^0.5;
%     fprintf("direction %d, gamma is %f\n",i, gamma(i));
% end

iter = (1:iter)';
plot(log(iter),log10(err),'.');
hold on 
plot(log(iter),-1*log10(iter)-0.6,'-.'); % k=0 
%plot(log(iter),-2*log10(iter)-1.5,'-.'); % k=1
st = 7;
temp = polyfit(log(iter(st:end)),log(err(st:end)),1);
fprintf('The convergence rate is %.2e \n', -temp(1));
%--------------------------------------------------------------------------
function z = epnorm(F)
z = 5/18*sum( F(1:3:end) )+4/9*sum( F(2:3:end) )+5/18*sum( F(3:3:end) );
end
%--------------------------------------------------------------------------
function [c,ceq] = mycon(x,G,h)
c = [];
ceq = h*epnorm((G*x).^2) - 1;
end
%--------------------------------------------------------------------------
function y = eqi(x,G,F,h,p)
i = size(G,2);
y = zeros(i,1);
r = F - G*x;
for k=1:i
    y(i) = h*epnorm(abs(r).^(p-2).*r.*G(:,i));
end
end



