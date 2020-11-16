/******************************
 * Author: sharkdtu
 * Date: 2016-10-29
 *****************************/

#ifndef TDW_H_
#define TDW_H_

/* All APIs set errno to meaningful values */

#ifdef __cplusplus
extern  "C" {
#endif

/**
 * The C reflection of com.tencent.tdw.common.TDWClient .
 */
struct tdw_client_internal;
typedef struct tdw_client_internal* tdw_client_t;

/**
 * The C reflection of com.tencent.tdw.common.io.TDWRecordReader .
 */
struct tdw_record_reader_internal;
typedef struct tdw_record_reader_internal* tdw_record_reader_t;

/**
 * The C reflection of com.tencent.tdw.common.io.TDWRecordWriter .
 */
struct tdw_record_writer_internal;
typedef struct tdw_record_writer_internal* tdw_record_writer_t;

/** 
 * New a client for accessing tdw.
 * 
 * @param db         The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param user       The port on which the server is listening.
 * @param password   The port on which the server is listening.
 * @return           A client handle to the tdw.
 */
tdw_client_t tdw_new_client(const char* db,
                            const char* user,
                            const char* password);

/** 
 * New a client for accessing tdw.
 * 
 * @param db         The NameNode.  See hdfsBuilderSetNameNode for details.
 * @param user       The port on which the server is listening.
 * @param password   The port on which the server is listening.
 * @param group      The cluster name, e.g. tl, cft, hk.
 * @return           A client handle to the tdw.
 *                   On error, errno will be set appropriately.
 */
tdw_client_t tdw_new_client_of_group(const char* db,
                                     const char* user,
                                     const char* password,
                                     const char* group);

/** 
 * Free the tdw client.
 *
 * @param tdw   The tdw client to be freed.
 * @return      Returns 0 on success, -1 on error.
 *              On error, errno will be set appropriately.
 */
int tdw_free_client(tdw_client_t tdw);
    
/** 
 * Get a tdw record reader.
 *
 * @param tdw   The tdw client.
 * @param path  The full path to the data file.
 * @return      Returns a tdw record reader to the path.
 *              On error, errno will be set appropriately.
 */
tdw_record_reader_t tdw_get_record_reader(tdw_client_t tdw, const char* path);

/** 
 * Close a record reader. 
 *
 * @param tdw   The tdw record reader to be closed.
 * @return      Returns 0 on success, -1 on error.  
 *              On error, errno will be set appropriately.
 */
int tdw_close_record_reader(tdw_record_reader_t reader);

/** 
 * Read the next record.
 *
 * @param reader  The record reader for reading.
 * @param record  (out param) The result record.
 * @return        Returns 0 on success, 1 on eof, -1 on error.  
 *                On error, errno will be set appropriately.
 */
int tdw_read_next(tdw_record_reader_t reader, char** record);

/** 
 * Free the record.
 *
 * @param record  The record readed by a tdw record reader.
 * @return 
 */
void tdw_free_record(char* record);


/** 
 * Get data paths for the specific table/partitions. 
 *
 * @param tdw        The tdw client.
 * @param table      The table name.
 * @param pri_parts  (option) The comma separated pri partition names.
 * @param sub_parts  (option) The comma separated sub partition names.
 * @param paths      (out param) The comma separated result data paths.
 * @return           Returns 0 on success, -1 on error.
 *                   On error, errno will be set appropriately .  
 */
int tdw_get_data_paths(tdw_client_t tdw,
                       const char* table,
                       const char* pri_parts,
                       const char* sub_parts, 
                       char** paths); 

/** 
 * Free data paths. 
 *
 * @param paths      The data paths to be freed.  
 * @return
 */
void tdw_free_data_paths(char* paths); 

/** 
 * Get a tdw record writer.
 *
 * @param tdw        The tdw client.
 * @param table      The table to be written.
 * @param pri_part   (option)The pri partition to be written.
 * @param sub_part   (option)The sub partition to be written.
 * @return           Returns a tdw record writer.
 *                   On error, errno will be set appropriately.
 */
tdw_record_writer_t tdw_get_record_writer(tdw_client_t tdw,
                                          const char* table,
                                          const char* pri_part,
                                          const char* sub_part);

/** 
 * Write a record.
 *
 * @param writer  The record writer for writing.
 * @param record  The record to be written.
 * @return        Returns 0 on success, -1 on error.  
 *                On error, errno will be set appropriately.
 */
int tdw_write_record(tdw_record_writer_t writer, const char* record);

/** 
 * Close a record writer. 
 *
 * @param tdw   The tdw record writer to be closed.
 * @return      Returns 0 on success, -1 on error.  
 *              On error, errno will be set appropriately.
 */
int tdw_close_record_writer(tdw_record_writer_t writer);

#ifdef __cplusplus
}
#endif

#endif /* TDW_H_ */
