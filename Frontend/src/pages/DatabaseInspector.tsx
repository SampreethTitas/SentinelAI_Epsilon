import React, { useState } from 'react';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../components/ui/accordion';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import { ScrollArea } from '../components/ui/scroll-area';
import LoadingShimmer from '../components/LoadingShimmer';
import { Eye, Database, Table as TableIcon, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react';

interface TableInfo {
    table_name: string;
    row_count: number;
    columns: Array<{
        name: string;
        type: string;
        nullable: boolean;
        default: string | null;
        max_length: number | null;
    }>;
}

interface DatabaseInfo {
    success: boolean;
    message: string;
    database_name: string;
    tables: TableInfo[];
    timestamp: string;
}

interface TableContent {
    success: boolean;
    message: string;
    table_name: string;
    columns: string[];
    rows: any[][];
    total_rows: number;
    returned_rows: number;
    timestamp: string;
}

const DatabaseInspector: React.FC = () => {
    const [connectionString, setConnectionString] = useState('');
    const [databaseInfo, setDatabaseInfo] = useState<DatabaseInfo | null>(null);
    const [selectedTable, setSelectedTable] = useState<string | null>(null);
    const [tableContent, setTableContent] = useState<TableContent | null>(null);
    const [loading, setLoading] = useState(false);
    const [contentLoading, setContentLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Pagination states
    const [currentPage, setCurrentPage] = useState(1);
    const [rowsPerPage] = useState(100);

    const API_BASE_URL = process.env.NODE_ENV === 'production'
        ? 'https://api.yourdomain.com'
        : 'http://localhost:8002';

    const inspectDatabase = async () => {
        if (!connectionString.trim()) {
            setError('Please enter a connection string');
            return;
        }

        setLoading(true);
        setError(null);
        setDatabaseInfo(null);
        setSelectedTable(null);
        setTableContent(null);

        try {
            const response = await fetch(`${API_BASE_URL}/datavalut/inspect-database`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    connection_string: connectionString
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail?.message || 'Failed to inspect database');
            }

            const data: DatabaseInfo = await response.json();
            setDatabaseInfo(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    const loadTableContent = async (tableName: string, page: number = 1) => {
        setContentLoading(true);
        setError(null);

        const offset = (page - 1) * rowsPerPage;

        try {
            const response = await fetch(`${API_BASE_URL}/datavalut/table-content`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    connection_string: connectionString,
                    table_name: tableName,
                    limit: rowsPerPage,
                    offset: offset
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail?.message || 'Failed to load table content');
            }

            const data: TableContent = await response.json();
            setTableContent(data);
            setSelectedTable(tableName);
            setCurrentPage(page);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setContentLoading(false);
        }
    };

    const getColumnTypeColor = (type: string) => {
        const typeColors: { [key: string]: string } = {
            'integer': 'bg-blue-100 text-blue-800',
            'bigint': 'bg-blue-100 text-blue-800',
            'character varying': 'bg-green-100 text-green-800',
            'text': 'bg-green-100 text-green-800',
            'boolean': 'bg-purple-100 text-purple-800',
            'timestamp': 'bg-orange-100 text-orange-800',
            'date': 'bg-orange-100 text-orange-800',
            'numeric': 'bg-yellow-100 text-yellow-800',
            'real': 'bg-yellow-100 text-yellow-800',
            'uuid': 'bg-pink-100 text-pink-800',
        };

        return typeColors[type.toLowerCase()] || 'bg-gray-100 text-gray-800';
    };

    const totalPages = tableContent ? Math.ceil(tableContent.total_rows / rowsPerPage) : 0;

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex items-center space-x-2 mb-6">
                <Database className="h-6 w-6 text-blue-500" />
                <h1 className="text-3xl font-bold">Database Inspector</h1>
            </div>

            {/* Connection String Input */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                        <Database className="h-5 w-5" />
                        <span>Database Connection</span>
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex space-x-2">
                        <Input
                            type="text"
                            placeholder="postgresql://username:password@host:port/database"
                            value={connectionString}
                            onChange={(e) => setConnectionString(e.target.value)}
                            className="flex-1"
                        />
                        <Button
                            onClick={inspectDatabase}
                            disabled={loading}
                            className="min-w-[120px]"
                        >
                            {loading ? (
                                <>
                                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                    Inspecting...
                                </>
                            ) : (
                                <>
                                    <Eye className="mr-2 h-4 w-4" />
                                    Inspect
                                </>
                            )}
                        </Button>
                    </div>

                    {error && (
                        <Alert variant="destructive">
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}
                </CardContent>
            </Card>

            {/* Database Information */}
            {databaseInfo && (
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                                <Database className="h-5 w-5 text-green-500" />
                                <span>Database: {databaseInfo.database_name}</span>
                            </div>
                            <Badge variant="secondary">
                                {databaseInfo.tables.length} table{databaseInfo.tables.length !== 1 ? 's' : ''}
                            </Badge>
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <Accordion type="single" collapsible className="w-full">
                            {databaseInfo.tables.map((table, index) => (
                                <AccordionItem key={table.table_name} value={`table-${index}`}>
                                    <AccordionTrigger className="text-left">
                                        <div className="flex items-center justify-between w-full mr-4">
                                            <div className="flex items-center space-x-2">
                                                <TableIcon className="h-4 w-4" />
                                                <span className="font-medium">{table.table_name}</span>
                                            </div>
                                            <div className="flex space-x-2">
                                                <Badge variant="outline">
                                                    {table.row_count} row{table.row_count !== 1 ? 's' : ''}
                                                </Badge>
                                                <Badge variant="outline">
                                                    {table.columns.length} column{table.columns.length !== 1 ? 's' : ''}
                                                </Badge>
                                                <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        loadTableContent(table.table_name);
                                                    }}
                                                    className="h-6 px-2"
                                                >
                                                    <Eye className="h-3 w-3 mr-1" />
                                                    View Data
                                                </Button>
                                            </div>
                                        </div>
                                    </AccordionTrigger>
                                    <AccordionContent>
                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="font-medium mb-2">Columns</h4>
                                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                                                    {table.columns.map((column) => (
                                                        <div
                                                            key={column.name}
                                                            className="p-3 border rounded-lg space-y-1"
                                                        >
                                                            <div className="flex items-center space-x-2">
                                                                <span className="font-medium">{column.name}</span>
                                                                {!column.nullable && (
                                                                    <Badge variant="destructive" className="text-xs">
                                                                        NOT NULL
                                                                    </Badge>
                                                                )}
                                                            </div>
                                                            <Badge
                                                                variant="secondary"
                                                                className={`text-xs ${getColumnTypeColor(column.type)}`}
                                                            >
                                                                {column.type}
                                                                {column.max_length && `(${column.max_length})`}
                                                            </Badge>
                                                            {column.default && (
                                                                <div className="text-xs text-gray-500">
                                                                    Default: {column.default}
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </AccordionContent>
                                </AccordionItem>
                            ))}
                        </Accordion>
                    </CardContent>
                </Card>
            )}

            {/* Table Content */}
            {selectedTable && tableContent && (
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                                <TableIcon className="h-5 w-5 text-blue-500" />
                                <span>Table: {selectedTable}</span>
                            </div>
                            <div className="flex items-center space-x-2">
                                <Badge variant="secondary">
                                    Showing {tableContent.returned_rows} of {tableContent.total_rows} rows
                                </Badge>
                                {totalPages > 1 && (
                                    <div className="flex items-center space-x-2">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => loadTableContent(selectedTable, currentPage - 1)}
                                            disabled={currentPage === 1 || contentLoading}
                                        >
                                            <ChevronLeft className="h-4 w-4" />
                                        </Button>
                                        <span className="text-sm">
                                            Page {currentPage} of {totalPages}
                                        </span>
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => loadTableContent(selectedTable, currentPage + 1)}
                                            disabled={currentPage === totalPages || contentLoading}
                                        >
                                            <ChevronRight className="h-4 w-4" />
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        {contentLoading ? (
                            <LoadingShimmer />
                        ) : (
                            <ScrollArea className="h-[400px] w-full">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            {tableContent.columns.map((column) => (
                                                <TableHead key={column} className="font-medium">
                                                    {column}
                                                </TableHead>
                                            ))}
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {tableContent.rows.map((row, rowIndex) => (
                                            <TableRow key={rowIndex}>
                                                {row.map((cell, cellIndex) => (
                                                    <TableCell key={cellIndex} className="max-w-xs truncate">
                                                        {cell === null ? (
                                                            <span className="text-gray-400 italic">NULL</span>
                                                        ) : (
                                                            String(cell)
                                                        )}
                                                    </TableCell>
                                                ))}
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </ScrollArea>
                        )}
                    </CardContent>
                </Card>
            )}
        </div>
    );
};

export default DatabaseInspector;
