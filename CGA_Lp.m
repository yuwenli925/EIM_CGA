%% Approximate the function f on [0,1] by CGA iterations
iter = 50; 
f = @(z) sin(pi*z); 
N = 500; h = 1/N;
node = (0:h:1)';

c = [1/2-sqrt(15)/10 1/2 1/2+sqrt(15)/10];% 3-point Gauss quadrature
qpt = [node(1:end-1)+c(1)*h;node(1:end-1)+c(2)*h;node(1:end-1)+c(3)*h];

hd = 5e-5; b = (-2.0:hd:2.0)'; % discretization of the dictionary
nd = length(b); 

g = [repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b']>0; % ReLU0

err = zeros(iter,3); 
for p = [2.5, 3, 3.5]
    ip = (p - 2.5)/0.5 + 1;
    fqpt = f(qpt); r = fqpt;
    A = zeros(iter,iter); rhs = zeros(iter,1); % matrix and rhs for projection
    id = zeros(iter,1); argmax = zeros(nd,1);
    C = 0;
    for i = 1:iter
        for j = 1:nd
            argmax(j) = epnorm( g(:,j).*abs(r).^(p-2).*r ); % norming 2
        end
        [~,id(i)] = max(abs(argmax));

        obj = @(x) epnorm(abs(fqpt - g(:,id(1:i))*x).^p);
        C = fminunc(obj,zeros(i,1));
        r = fqpt - g(:,id(1:i))*C;
        C = [C;0];
        err(i,ip) = (h*epnorm(abs(r).^p)).^(1/p);

        fprintf("Step %d, error in L%0.1f is %f\n",i,p,err(i,ip));
    end
end

%% Compute the order of convergence of the CGA for ReLU0

iter = (1:iter)';
 
st = 7;

temp = polyfit(log(iter(st:end)),log(err(st:end,1)),1);
fprintf('Convergence order in L2.5 is %.2e \n', -temp(1));

temp = polyfit(log(iter(st:end)),log(err(st:end,2)),1);
fprintf('Convergence order in L3 is %.2e \n', -temp(1));

temp = polyfit(log(iter(st:end)),log(err(st:end,3)),1);
fprintf('Convergence order in L3.5 is %.2e \n', -temp(1));

%% Plot the order of convergence of the CGA for ReLU0

plot(log10(iter),log10(err(:,1)),'-o');
hold on 
plot(log10(iter),log10(err(:,2)),'-*');
plot(log10(iter),log10(err(:,3)),'-d');
plot(log10(iter),-1*log10(iter)-0.6,'-.');
axis tight
%--------------------------------------------------------------------------
function z = epnorm(F)
%z = 5/18*sum( F(1:3:end) )+4/9*sum( F(2:3:end) )+5/18*sum( F(3:3:end) );
n = length(F)/3;
z = 5/18*sum( F(1:n) )+4/9*sum( F(n+1:2*n) )+5/18*sum( F(2*n+1:3*n) );
end
%--------------------------------------------------------------------------



