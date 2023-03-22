@echo off
%获取当前路径%
echo %~dp0
%执行exe%
.\app\imageSim.exe -s %~dp0\data\sample\sample4.png -q %~dp0\data\query\ -o %~dp0\output\