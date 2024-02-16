function R = quantify_policy(R)
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2024. February 08. (2023a)
%

R_ = join(join(join(join(join(join(R(:,Vn.policy),Vn.TP),Vn.PL),Vn.CF),Vn.SO),Vn.QU),Vn.MA);

for vn = Vn.policy + "_Val"
    R.(vn) = R_.(vn);
end

R.Iq = Vn.Iq(R);

end