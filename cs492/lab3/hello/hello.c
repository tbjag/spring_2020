//  
//  CS492 - Lab #3 part 1
//
//  Student: Theodore Jagodits
//

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>

//  Define the module metadata.
#define MODULE_NAME "hello"
MODULE_AUTHOR("THEODORE JAGODITS");
MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("A simple kernel module to greet the installer");
MODULE_VERSION("0.1");

//  Define the name parameter.
static char *name = "tjagodit";

module_param(name, charp, S_IRUGO);
MODULE_PARM_DESC(name, "The name to display in /var/log/kern.log");

static int __init hello_init(void)
{
    pr_info("%s: module loaded at 0x%p\n", MODULE_NAME, hello_init);
    pr_info("Hello user %s\n", name);
    return 0;
}

static void __exit hello_exit(void)
{
    pr_info("Goodbye user %s\n", name);
    pr_info("%s: module unloaded from 0x%p\n", MODULE_NAME, hello_exit);
}

// register the operations to be executed when the KM is loaded and unloaded
module_init(hello_init);
module_exit(hello_exit);