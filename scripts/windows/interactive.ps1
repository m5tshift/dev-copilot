# Runs a .cmd file in the user's interactive logon session and waits for it.
#
# Why: a process started over SSH dies with the SSH session, and Docker's credential
# helper and Docker Sandboxes (sbx) need the interactive logon session (Credential
# Manager): from SSH they fail with "logon session does not exist". A scheduled task
# with LogonType Interactive runs in the session of the logged-on user. No trigger:
# it runs only when started.
#
#   powershell -ExecutionPolicy Bypass -File interactive.ps1 C:\src\job.cmd [-Name job] [-NoWait]
param([Parameter(Mandatory)][string]$Script, [string]$Name = "devcopilot-job", [switch]$NoWait)
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$Script`""
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
Register-ScheduledTask -TaskName $Name -Action $action -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $Name
if ($NoWait) { "started $Name"; exit }
Start-Sleep 2
while ((Get-ScheduledTask -TaskName $Name).State -eq "Running") { Start-Sleep 2 }
