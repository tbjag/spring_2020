//  Note:
//  This code is based on Derek Molloy's excellent tutorial on creating Linux
//  Kernel Modules:
//
//  CS 492 - Lab 3
//  Student: Theodore Jagodits
//
//    http://derekmolloy.ie/writing-a-linux-kernel-module-part-2-a-character-device/
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
//  Define the module metadata.
#define MODULE_NAME "qmod"
#define DEVICE_NAME "qmod"
#define CLASS_NAME  "qmod"
#define BUFFER_SIZE  256

MODULE_AUTHOR("Theodore Jagodits");
MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("A module which provides a shared queue");
MODULE_VERSION("0.1");

//  Define the name parameter.
static char *name = "tjagodit";
module_param(name, charp, S_IRUGO);
MODULE_PARM_DESC(name, "The name to display in /var/log/kern.log");

//  The device number, automatically set. The message buffer and current message
//  size. The number of device opens and the device class struct pointers.
static int    majorNumber;
static struct class*  qmodClass  = NULL;
static struct device* qmodDevice = NULL;

// Declare the shared queue data structure (a static data structure in the KM).
#define BUFFER_LENGTH 256

struct qnode {
  char message[BUFFER_LENGTH];
  int mesglen;
  struct qnode *next;
};

struct qheader {
  struct qnode *front;
  struct qnode *last;
};

static struct qheader qHeader;
// Declare a mutex for ensuring mutual exclusion on the read and write operations.
DEFINE_MUTEX(sem);
  
//  Prototypes for our device functions.
static int     dev_open(struct inode *, struct file *);
static int     dev_release(struct inode *, struct file *);
static ssize_t dev_read(struct file *, char *, size_t, loff_t *);
static ssize_t dev_write(struct file *, const char *, size_t, loff_t *);


//  Create the file operations instance for our driver.
static struct file_operations fops =
{
   .open = dev_open,
   .read = dev_read,
   .write = dev_write,
   .release = dev_release,
};

static int __init mod_init(void)
{
    pr_info("%s: module loaded at 0x%p\n", MODULE_NAME, mod_init);

    // Initialize the queue data structure
    qHeader.front = NULL;
    qHeader.last = NULL;
    // Create a mutex to guard io operations.
    mutex_init(&sem);
    
    //  Register the device, allocating a major number.
    majorNumber = register_chrdev(0 /* i.e. allocate a major number for me */, DEVICE_NAME, &fops);
    if (majorNumber < 0) {
        pr_alert("%s: failed to register a major number\n", MODULE_NAME);
        return majorNumber;
    } 
    pr_info("%s: registered correctly with major number %d\n", MODULE_NAME, majorNumber);

    //  Create the device class.
    qmodClass = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(qmodClass)) {
        //  Cleanup resources and fail.
        unregister_chrdev(majorNumber, DEVICE_NAME);
        pr_alert("%s: failed to register device class\n", MODULE_NAME);

        //  Get the error code from the pointer.
        return PTR_ERR(qmodClass);
    }
    pr_info("%s: device class registered correctly\n", MODULE_NAME);

    //  Create the device.
    qmodDevice = device_create(qmodClass, NULL, MKDEV(majorNumber, 0), NULL, DEVICE_NAME);
    if (IS_ERR(qmodDevice)) {
        class_destroy(qmodClass);
        unregister_chrdev(majorNumber, DEVICE_NAME);
        pr_alert("%s: failed to create the device\n", DEVICE_NAME);
        return PTR_ERR(qmodDevice);
    }
    pr_info("%s: device class created correctly\n", DEVICE_NAME);

    //  Success!
    return 0;
}

static void __exit mod_exit(void)
{
    pr_info("%s: unloading...\n", MODULE_NAME);
    device_destroy(qmodClass, MKDEV(majorNumber, 0));
    class_unregister(qmodClass);
    class_destroy(qmodClass);
    unregister_chrdev(majorNumber, DEVICE_NAME);

    // destroy the queue
    mutex_lock(&sem);
    while(qHeader.front != NULL){
	struct qnode *temp = qHeader.front;
	qHeader.front = temp->next;
	kfree(temp);
    }
    qHeader.last = NULL;
    // destroy mutex
    mutex_unlock(&sem);
    mutex_destroy(&sem);

    pr_info("%s: device unregistered\n", MODULE_NAME);
}
 
/** @brief The device open function that is called each time the device is opened
 *  @param inodep A pointer to an inode object (defined in linux/fs.h)
 *  @param filep A pointer to a file object (defined in linux/fs.h)
 */
static int dev_open(struct inode *inodep, struct file *filep){
    pr_info("%s: device has been opened.\n", MODULE_NAME);
    return 0;
}

/** @brief This function is called whenever device is being read from user space i.e. data is
 *  being sent from the device to the user. In this case is uses the copy_to_user() function to
 *  send the buffer string to the user and captures any errors.
 *  @param filep A pointer to a file object (defined in linux/fs.h)
 *  @param buffer The pointer to the buffer to which this function writes the data
 *  @param len The length of the b
 *  @param offset The offset if required
 */
static ssize_t dev_read(struct file *filep, char *buffer, size_t len, loff_t *offset){
   char message[BUFFER_LENGTH] = { 0 };
   int error_count;
   struct qnode *qNode;

   // Use a mutex to avoid race conditions.
   mutex_lock(&sem);
   qNode = qHeader.front;
   // remove a message from the queue, return -1 if the queue is empty
   // Use a mutex to avoid race conditions.
   if(qNode == NULL){
     mutex_unlock(&sem);
     return -1;
   }
   if (qNode->mesglen > len) {
     mutex_unlock(&sem);
     pr_err("%s: input lenghth too small (error=%d)\n", MODULE_NAME, -EFAULT);
     return -EFAULT;              
   }

   //fill buffer
   len = qNode->mesglen;
   memcpy(message, qNode->message, len);
   //change nodes
   if (qNode->next == NULL){
      qHeader.front = NULL;
      qHeader.last = NULL;
   } else {
      qHeader.front = qNode->next;
   }
   mutex_unlock(&sem);
   //free current node
   kfree(qNode);
    
   // copy_to_user has the format ( * to, *from, size) and returns 0 on success
   error_count = copy_to_user(buffer, message, len);
   
   if (error_count==0) {
      pr_info("%s: sent %d characters to a client\n", MODULE_NAME, error_count);
      return 0;
   }
   else {
      pr_err("%s: failed to send message to a client (error=%d)\n", MODULE_NAME, error_count);
      //    Failed -- return a bad address message (i.e. -14)
      return -EFAULT;              
   }
}

/** @brief This function is called whenever the device is being written to from user space i.e.
 *  data is sent to the device from the user. The data is copied to the message[] array in this
 *  LKM using the sprintf() function along with the length of the string.
 *  @param filep A pointer to a file object
 *  @param buffer The buffer to that contains the string to write to the device
 *  @param len The length of the array of data that is being passed in the const char buffer
 *  @param offset The offset if required
 */
static ssize_t dev_write(struct file *filep, const char *buffer, size_t len, loff_t *offset){
    // Add the message and its length to the queue
    struct qnode * qNode;
    // Use a mutex to avoid race conditions.
    mutex_lock(&sem);
    // kmalloc new node
    qNode = kmalloc(sizeof(struct qnode), GFP_USER);
    // if queue empty
    if (qNode == NULL){
        mutex_unlock(&sem);
	return -1;
    }
    //if len is greater than buffer size
    if (BUFFER_SIZE < len) {
        len = BUFFER_SIZE;
    }
    // copy over from buffer to node
    memcpy(qNode->message ,buffer, len* sizeof(char));
    // insert in queue
    qNode->next = NULL;
    qNode->mesglen = len;
    if (qHeader.front == NULL){
	qHeader.front = qNode;
	qHeader.last = qNode;
    } else{
	qHeader.last->next = qNode;
	qHeader.last = qNode;
    }
    
    mutex_unlock(&sem);
    pr_info("%s: received %zu characters from the user\n", MODULE_NAME, len);
    return len;
} 

/** @brief The device release function that is called whenever the device is closed/released by
 *  the userspace program
 *  @param inodep A pointer to an inode object (defined in linux/fs.h)
 *  @param filep A pointer to a file object (defined in linux/fs.h)
 */
static int dev_release(struct inode *inodep, struct file *filep){
     pr_info("%s: device successfully closed\n", MODULE_NAME);
     return 0;
}


// register the operations to be executed when the KM is loaded and unloaded
module_init(mod_init);
module_exit(mod_exit);