taste <- c(12.3,20.9,39.0,47.9,5.6,25.9,37.3,21.9,18.1,21.0,34.9,57.2,0.7,25.9,54.9,40.9,15.9,6.4,18.0,38.9,14.0,15.2,32.0,56.7,16.8,11.6,26.5,0.7,13.4,5.5)
acetic <- c(4.543,5.159,5.366,5.759,4.663,5.697,5.892,6.078,4.898,5.242,5.740,6.446,4.477,5.236,6.151,6.365,4.787,5.412,5.247,5.438,4.564,5.298,5.455,5.855,5.366,6.043,6.458,5.328,5.802,6.176)
h2s <- c(3.135,5.043,5.438,7.496,3.807,7.601,8.726,7.966,3.850,4.174,6.142,7.908,2.996,4.942,6.752,9.588,3.912,4.700,6.174,9.064,4.949,5.220,9.242,10.199,3.664,3.219,6.962,3.912,6.685,4.787)
lactic <- c(0.86,1.53,1.57,1.81,0.99,1.09,1.29,1.78,1.29,1.58,1.68,1.90,1.06,1.30,1.52,1.74,1.16,1.49,1.63,1.99,1.15,1.33,1.44,2.01,1.31,1.46,1.72,1.25,1.08,1.25)

model.lm <- lm(taste~ acetic + h2s)
coef(summary(model.lm))
confint(model.lm,level=0.95)
anova(model.lm)
rstandard(model.lm)
summary(model.lm)
qqnorm(rstandard(model.lm))

densityplot(~taste, data = model.lm)

smallcheese = subset(model.lm, select = "taste", "acetic", "h2s", "lactic")
with(model.lm, cor(smallcheese))

pairs(smallcheese, pch=".")

plot(lm1, which = 2)

par(mfrow=c(1,2))
plot(residuals.lm(fm1),main="The 1st model")
abline(h=c(2,0,-2), col=c("blue", "red", "blue"), lty=c(2,1,2), lwd=c(1,3, 1))
plot(residuals.lm(fm2),main="The 2nd model")  
abline(h=c(2,0,-2), col=c("blue", "red", "blue"), lty=c(2,1,2), lwd=c(1,3, 1))


