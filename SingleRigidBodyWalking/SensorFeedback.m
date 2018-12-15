function SensorFeedback(dt,fc)
global FT
Ffilter=50.0;
Tfilter=1/Ffilter;
FT=(Tfilter*FT+dt*fc)./(Tfilter+dt);