#include <linux/sched.h>
#include <linux/syscalls.h>
#include <linux/kernel.h>
#include <asm/uaccess.h>

SYSCALL_DEFINE2(my_syscall, int, a, int, b)
{
	int c = a + b;
	printk(KERN_INFO "Task with pid %i running.\nThe sum of %d and %d is %d\n", current->pid, a, b, c);
       return c;	
}

SYSCALL_DEFINE1(my_syscall2, char*, input){
	int i = 0;
	int count = 0;
	long ret;
	char copy[128];
	printk(KERN_INFO "my_syscall2 pid is %i\n", current->pid);
	if(strlen(input) > 128){
		return -1;
	}else{
		ret = copy_from_user(copy, input, 128);
		while(copy[i] != '\0'){
			if(copy[i] == 'o'){
				copy[i] = '0';
				count++;
			}
			i++;
		}
		printk(KERN_INFO "The input is %s and the output is %s\n", input, copy);
		ret = copy_to_user(input, copy ,128);
		return count;
	}

}
