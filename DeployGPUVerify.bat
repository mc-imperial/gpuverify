@echo off

set INSTALL_DIR=C:\prog\GPUVerifyInstall

set LIBCLC_DIR=C:\prog\libclc
set BUGLE_DIR=C:\prog\bugle
set BUGLE_BIN_DIR=C:\prog\BugleBuild\Release
set LLVM_BUILD_DIR=C:\prog\llvm-build
set LLVM_BIN_DIR=C:\prog\llvm-build\bin\Release
set GPUVERIFY_DIR=C:\prog\GPUVerify
set GPUVERIFY_VCGEN_BIN_DIR=C:\prog\GPUVerify\GPUVerifyVCGen\bin\Debug
set Z3_BIN_DIR="C:\prog\GPUVerify\Binaries"

md %INSTALL_DIR%

xcopy /Y /E /I %LIBCLC_DIR% %INSTALL_DIR%\libclc


md %INSTALL_DIR%\bugle

xcopy /Y /E /I %BUGLE_DIR%\include-blang %INSTALL_DIR%\bugle\include-blang

xcopy /Y /E /I %LLVM_BUILD_DIR%\bin\lib %INSTALL_DIR%\lib


xcopy /Y %~dp0GPUVerify.py %INSTALL_DIR%
xcopy /Y %~dp0GPUVerify.bat %INSTALL_DIR%
xcopy /Y %~dp0GPUVerifyTester.py %INSTALL_DIR%

md %INSTALL_DIR%\bin

xcopy /Y %LLVM_BIN_DIR%\clang.exe %INSTALL_DIR%\bin
xcopy /Y %LLVM_BIN_DIR%\opt.exe %INSTALL_DIR%\bin
xcopy /Y %LLVM_BIN_DIR%\llvm-nm.exe %INSTALL_DIR%\bin
xcopy /Y %BUGLE_BIN_DIR%\bugle.exe %INSTALL_DIR%\bin

xcopy /Y %GPUVERIFY_DIR%\Binaries\AbsInt.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\Provers.SMTLib.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\Basetypes.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\CodeContractsExtender.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\Graph.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\VCExpr.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\Core.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\Houdini.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\VCGeneration.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\Model.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\ParserHelper.dll %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\UnivBackPred2.smt2 %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_DIR%\Binaries\GPUVerifyBoogieDriver.exe %INSTALL_DIR%\bin

xcopy /Y %GPUVERIFY_VCGEN_BIN_DIR%\GPUVerifyVCGen.exe %INSTALL_DIR%\bin
xcopy /Y %GPUVERIFY_VCGEN_BIN_DIR%\Predication.dll %INSTALL_DIR%\bin

xcopy /Y %Z3_BIN_DIR%\z3.exe %INSTALL_DIR%\bin
