; ModuleID = 'probe4.be4f8e02ff48993-cgu.0'
source_filename = "probe4.be4f8e02ff48993-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@alloc_771275e0c9485cd9e7235e6013545305 = private unnamed_addr constant <{ [64 x i8] }> <{ [64 x i8] c"/usr/src/debug/rust/rustc-1.75.0-src/library/core/src/num/mod.rs" }>, align 1
@alloc_88d116cdb4508f80368d04d7e97bdc2c = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_771275e0c9485cd9e7235e6013545305, [16 x i8] c"@\00\00\00\00\00\00\00y\04\00\00\05\00\00\00" }>, align 8
@str.0 = internal constant [25 x i8] c"attempt to divide by zero"

; probe4::probe
; Function Attrs: nonlazybind uwtable
define void @_ZN6probe45probe17h1884830d016d439bE() unnamed_addr #0 {
start:
  %0 = call i1 @llvm.expect.i1(i1 false, i1 false)
  br i1 %0, label %panic.i, label %"_ZN4core3num21_$LT$impl$u20$u32$GT$10div_euclid17h541e1c1f713d83d0E.exit"

panic.i:                                          ; preds = %start
; call core::panicking::panic
  call void @_ZN4core9panicking5panic17h56e581b6c5fac384E(ptr align 1 @str.0, i64 25, ptr align 8 @alloc_88d116cdb4508f80368d04d7e97bdc2c) #3
  unreachable

"_ZN4core3num21_$LT$impl$u20$u32$GT$10div_euclid17h541e1c1f713d83d0E.exit": ; preds = %start
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i1 @llvm.expect.i1(i1, i1) #1

; core::panicking::panic
; Function Attrs: cold noinline noreturn nonlazybind uwtable
declare void @_ZN4core9panicking5panic17h56e581b6c5fac384E(ptr align 1, i64, ptr align 8) unnamed_addr #2

attributes #0 = { nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { cold noinline noreturn nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #3 = { noreturn }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"RtLibUseGOT", i32 1}
!2 = !{!"rustc version 1.75.0 (82e1608df 2023-12-21) (Arch Linux rust 1:1.75.0-2)"}
